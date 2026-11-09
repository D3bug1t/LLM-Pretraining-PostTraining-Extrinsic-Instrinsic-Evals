"""AdaLoRA (Zhang et al. 2023, arXiv 2303.10512): LoRA whose rank is ADAPTED per layer during training.

IDEA: plain LoRA gives every layer the same rank r. But some layers matter more than others. AdaLoRA starts every
layer with a larger rank and, while training, PRUNES the least important rank-1 components, until only a target
total "budget" of components is left, shared across all layers - important layers keep more.

1. SVD-shaped update. Instead of B A, the update is written like a singular value decomposition:
       delta_W = P diag(lam) Q        P (out x r),  lam (r,),  Q (r x in)
   Component i is the "triplet" (column i of P, lam[i], row i of Q). Setting lam[i] = 0 removes it, without
   touching P and Q - so pruning is just a mask on lam. lam starts at 0 (so the layer starts as W0),
   P and Q start small random (std 0.02).
   A penalty  ||P^T P - I||^2 + ||Q Q^T - I||^2  (added to the loss) keeps P's columns and Q's rows orthonormal,
   so that lam really measures the size of each component, like singular values.

2. Importance of each triplet. The sensitivity of one parameter w is s = |w * dLoss/dw| (how much the loss would
   change if w were removed). Gradients are noisy, so AdaLoRA smooths it over steps (beta = 0.85):
       I_bar = beta1 * I_bar + (1 - beta1) * s                        (smoothed sensitivity)
       U_bar = beta2 * U_bar + (1 - beta2) * |s - I_bar|              (smoothed uncertainty)
       score = I_bar * U_bar
   Importance of triplet i = score(lam[i]) + mean of score over column i of P + mean of score over row i of Q.

3. Budget schedule. The total number of kept triplets (over ALL layers) goes from b0 = r_init * n_layers down to
   bT = r_target * n_layers with a cubic curve: no pruning during `warmup` steps, then
       b(t) = bT + (b0 - bT) * (1 - (t - warmup) / (total - final - warmup)) ** 3,
   and b = bT for the last `final` steps. Every few steps, keep the b(t) highest-scoring triplets, mask the rest.

Torch basics you will need:
  torch.randn(a, b) * 0.02                  -> small random matrix
  self.register_buffer("mask", torch.ones(r))  -> non-trained tensor saved with the module
  torch.eye(r, device=...)                  -> identity matrix
  (M ** 2).sum()                            -> squared Frobenius norm
  p.grad                                    -> gradient of a parameter after loss.backward() (None before)
  torch.zeros_like(t)                       -> zeros with t's shape/dtype/device
  t.mean(dim=0)                             -> mean over rows (one number per column)
  torch.cat([t1, t2])                       -> join 1-D tensors
  t.kthvalue(k).values                      -> the k-th SMALLEST value (k starts at 1)
  m.mask.copy_(new)                         -> overwrite a buffer in place
  @torch.no_grad()                          -> decorator: no autograd inside (bookkeeping code)
"""
import math
import torch
import torch.nn as nn


class AdaLoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        """r: the INITIAL rank (pruned later). Must give exactly base(x) before training."""
        super().__init__()
        # Step 1. self.base = base, frozen;  self.r = r;  self.scale = alpha / r
        # Step 2. self.P = Parameter (out_features, r), torch.randn * 0.02
        #         self.lam = Parameter (r,), zeros
        #         self.Q = Parameter (r, in_features), torch.randn * 0.02
        # Step 3. self.register_buffer("mask", torch.ones(r))      (1 = triplet kept, 0 = pruned)
        raise NotImplementedError

    def forward(self, x):
        # Step 1. frozen = self.base(x)
        # Step 2. z = x @ Q.t()                     -> (..., r)
        #         z = z * (lam * mask)              -> scales each component (pruned ones become 0)
        #         z = z @ P.t()                     -> (..., out)
        # Step 3. return frozen + scale * z
        raise NotImplementedError

    def orth_penalty(self):
        """||P^T P - I||^2 + ||Q Q^T - I||^2 (both r x r). Added to the loss as  loss + gamma * sum over layers."""
        # Step 1. I = torch.eye(self.r, device=self.P.device)
        # Step 2. return ((P.t() @ P - I) ** 2).sum() + ((Q @ Q.t() - I) ** 2).sum()
        raise NotImplementedError


class RankAllocator:
    """Bookkeeping for the pruning. Call, every training step, AFTER loss.backward() and BEFORE optimizer.step():
           alloc.update_scores();  if step % every == 0: alloc.mask_to_budget(step)"""
    def __init__(self, layers, init_r, target_r, total_steps, warmup, final, beta1=0.85, beta2=0.85):
        """layers: dict name -> AdaLoRALinear."""
        # Step 1. Store everything as attributes (self.layers = layers, self.warmup = warmup, ...).
        # Step 2. self.b0 = init_r * len(layers);  self.bT = target_r * len(layers)
        # Step 3. self.ipt = {} and self.unc = {}  (smoothed sensitivity / uncertainty, keyed by (name, "P"/"lam"/"Q"))
        raise NotImplementedError

    def budget(self, step):
        """Total number of triplets to keep at this step (an int), following the cubic schedule above."""
        # Step 1. if step < warmup: return b0
        # Step 2. if step >= total_steps - final: return bT
        # Step 3. frac = 1 - (step - warmup) / (total_steps - final - warmup)
        #         return int(bT + (b0 - bT) * frac ** 3)
        raise NotImplementedError

    @torch.no_grad()
    def update_scores(self):
        """Update ipt / unc of every P, lam, Q from their current .grad."""
        # For every name, m in self.layers.items(), and every pname in ("P", "lam", "Q"):
        # Step 1. p = getattr(m, pname); skip if p.grad is None.
        # Step 2. s = (p * p.grad).abs();   key = (name, pname)
        # Step 3. First time this key is seen: self.ipt[key] = zeros_like(s), self.unc[key] = zeros_like(s).
        # Step 4. new_ipt = beta1 * ipt[key] + (1 - beta1) * s
        #         unc[key] = beta2 * unc[key] + (1 - beta2) * (s - new_ipt).abs()
        #         ipt[key] = new_ipt
        raise NotImplementedError

    def triplet_scores(self, name):
        """(r,) importance of each triplet of layer `name` (assumes update_scores has run at least once)."""
        # Step 1. S = {pname: ipt[(name, pname)] * unc[(name, pname)] for pname in ("P", "lam", "Q")}
        # Step 2. return S["lam"] + S["P"].mean(0) + S["Q"].mean(1)
        #         (P is (out x r): mean over dim 0 -> one number per column i.  Q is (r x in): mean over dim 1.)
        raise NotImplementedError

    @torch.no_grad()
    def mask_to_budget(self, step):
        """Keep the budget(step) best triplets over ALL layers, set the others' mask to 0. Returns the budget."""
        # Step 1. b = self.budget(step);  scores = {name: self.triplet_scores(name) for name in self.layers}
        # Step 2. everything = torch.cat(list(scores.values()));  n = everything.numel()
        # Step 3. If b >= n: set every mask to ones and return b.
        # Step 4. thresh = everything.kthvalue(n - b).values      (the (n-b)-th smallest; strictly above it = top b)
        # Step 5. For each layer: m.mask.copy_((scores[name] > thresh).float())
        # Step 6. return b
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(16, 12)
    m = AdaLoRALinear(base, r=6, alpha=12)
    x = torch.randn(3, 16)
    print(torch.allclose(m(x), base(x)))                                  # expect True (lam starts at 0)
    print(sorted(n for n, p in m.named_parameters() if p.requires_grad))  # expect ['P', 'Q', 'lam']

    m.P.data = torch.linalg.qr(torch.randn(12, 6))[0]                     # orthonormal columns
    m.Q.data = torch.linalg.qr(torch.randn(16, 6))[0].t()                 # orthonormal rows
    print(round(m.orth_penalty().item(), 6))                              # expect 0.0

    layers = {"a": AdaLoRALinear(nn.Linear(16, 12), 8, 16), "b": AdaLoRALinear(nn.Linear(16, 12), 8, 16)}
    alloc = RankAllocator(layers, init_r=8, target_r=4, total_steps=100, warmup=10, final=20)
    print([alloc.budget(t) for t in (0, 10, 40, 79, 80, 99)])             # expect [16, 16, 9, 8, 8, 8]
    for step in range(100):
        loss = sum((l(torch.randn(4, 16)) ** 2).mean() for l in layers.values())
        for l in layers.values(): l.lam.data.add_(0.01)                   # pretend training moved lam away from 0
        for l in layers.values(): l.zero_grad()
        loss.backward()
        alloc.update_scores()
        if step % 5 == 0: alloc.mask_to_budget(step)
    alloc.mask_to_budget(99)
    print(int(sum(l.mask.sum() for l in layers.values())))                # expect 8 (= target_r * 2 layers)
    print({k: int(l.mask.sum()) for k, l in layers.items()})              # the 8 kept, split between the layers
