"""QLoRA (Dettmers et al. 2023, arXiv 2305.14314): LoRA on top of a 4-bit QUANTISED frozen base weight.

IDEA: in LoRA the base weight W0 is frozen, so we never need its full precision for training - only to compute W0 x.
QLoRA stores W0 in 4 bits (16 possible values per number instead of ~4 billion) and rebuilds ("dequantises")
an approximate W0 on the fly in every forward pass. The LoRA matrices A, B stay in full precision and are trained
as usual; gradients flow through the dequantised weight into the input, but the 4-bit codes never change.

How 4-bit "NormalFloat" (NF4) quantisation works:
  1. Flatten W0 and cut it into BLOCKS of 64 numbers.
  2. For each block, absmax = the largest |value| in it. Divide the block by absmax -> every number is in [-1, 1].
  3. Replace each number by the index (0..15) of the NEAREST of 16 fixed levels, NF4_LEVELS below.
     The levels are not evenly spaced: they are quantiles of a normal distribution, because trained weights are
     roughly normally distributed - more levels near 0, where most weights are.
  4. Store: the 4-bit codes + one absmax per block.  Dequantise:  value = NF4_LEVELS[code] * absmax_of_its_block.
(The real QLoRA also packs two 4-bit codes into one byte and quantises the absmax values too ("double
quantisation"); we keep one code per uint8 byte, which is simpler and still 4x smaller than float32.)

Torch basics you will need:
  W.detach().float().reshape(-1)          -> a flat float32 copy, not connected to autograd
  F.pad(t, (0, k))                        -> append k zeros at the end of a 1-D tensor
  t.view(-1, 64)                          -> reshape into rows of 64 (number of rows is inferred)
  t.abs().amax(dim=1)                     -> max of |t| along each row
  t.clamp(min=1e-12)                      -> avoid dividing by 0 for an all-zero block
  torch.bucketize(x, boundaries)          -> for each x, the index of the interval of `boundaries` it falls in
  t.to(torch.uint8), t.long()             -> change dtype (indexing needs long)
  LEVELS[codes]                           -> look up a value for each code (fancy indexing)
  self.register_buffer("name", tensor)    -> a tensor saved with the module and moved by .to(device), but NOT trained
  math.prod(shape)                        -> number of elements of a shape tuple
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0])


def quantize_nf4(W, block=64):
    """W: any float tensor. Returns (codes, absmax):
    codes  uint8 tensor (n_blocks, block) with values 0..15,   absmax float32 tensor (n_blocks,)."""
    # Step 1. flat = a detached float32 1-D copy of W.
    # Step 2. Pad flat with zeros so its length is a multiple of block: pad = (-flat.numel()) % block.
    # Step 3. blocks = flat.view(-1, block);  absmax = max |value| of each row, clamped to >= 1e-12.
    # Step 4. normed = blocks / absmax[:, None]        (the [:, None] makes absmax a column so it divides each row)
    # Step 5. Nearest level: the boundaries between neighbouring levels are their midpoints,
    #         mid = (NF4_LEVELS[1:] + NF4_LEVELS[:-1]) / 2   (15 numbers);  codes = torch.bucketize(normed, mid).
    #         (bucketize returns 0 for values below mid[0], 15 for values above mid[14]: exactly the nearest level.)
    # Step 6. return codes as uint8, absmax
    raise NotImplementedError


def dequantize_nf4(codes, absmax, shape):
    """Inverse of quantize_nf4: returns a float32 tensor of the given shape (approximately the original W)."""
    # Step 1. vals = NF4_LEVELS (moved to absmax.device) indexed by codes.long()   -> (n_blocks, block)
    # Step 2. vals = vals * absmax[:, None]
    # Step 3. Flatten, drop the padding (keep the first math.prod(shape) numbers), reshape to shape.
    raise NotImplementedError


class QLoRALinear(nn.Module):
    def __init__(self, base, r, alpha, block=64):
        """base: a pretrained nn.Linear. Its weight is quantised and then NOT kept (that is the memory saving).
        The wrapped layer must give approximately base(x) before training (not exactly: quantisation error)."""
        super().__init__()
        # Step 1. codes, absmax = quantize_nf4(base.weight, block); store both with self.register_buffer(...).
        #         Also remember self.shape = tuple(base.weight.shape).
        # Step 2. Bias: if base.bias is None keep self.bias = None, else a FROZEN copy:
        #         nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        # Step 3. LoRA part, exactly as in lora.py: self.scale = alpha / r,
        #         self.A (r, in_features) kaiming_uniform_, self.B (out_features, r) zeros.
        #         (Do NOT store `base` itself: then its full-precision weight would stay in memory.)
        raise NotImplementedError

    def forward(self, x):
        # Step 1. W = dequantize_nf4(self.codes, self.absmax, self.shape).to(x.dtype)
        # Step 2. frozen = F.linear(x, W, self.bias)
        # Step 3. return frozen + scale * (x @ A.t()) @ B.t()       (the LoRA path, as in lora.py)
        raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    W = torch.randn(300, 200) * 0.02
    codes, absmax = quantize_nf4(W)
    print(codes.dtype, codes.shape, int(codes.min()), int(codes.max()), absmax.shape)
    # expect torch.uint8 torch.Size([938, 64]) 0 15 torch.Size([938])
    W2 = dequantize_nf4(codes, absmax, W.shape)
    print(W2.shape, round(((W - W2).norm() / W.norm()).item(), 3))
    # expect torch.Size([300, 200]) and a relative error around 0.1 (4 bits are not exact)

    base = nn.Linear(200, 300)
    q = QLoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, 200)
    print(round(((q(x) - base(x)).norm() / base(x).norm()).item(), 3))     # expect small, around 0.1
    print([n for n, p in q.named_parameters() if p.requires_grad])          # expect ['A', 'B']
    print(sorted(n for n, _ in q.named_buffers()))                          # expect ['absmax', 'codes']
    q.B.data.normal_(); q(x).sum().backward()
    print(q.A.grad is not None, q.B.grad is not None)                       # expect True True
