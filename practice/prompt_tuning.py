"""Prompt Tuning (Lester, Al-Rfou & Constant 2021, arXiv 2104.08691): learn a few "virtual token" embeddings.

IDEA: freeze the WHOLE model. Prepend n learned vectors ("soft prompt") to the token embeddings of every input:
        embeddings (B, T, d)   ->   [prompt_1 .. prompt_n, tok_1 .. tok_T]   (B, n + T, d)
The model reads them like ordinary tokens. Only the n x d prompt matrix (plus the task head) is trained.
The paper initialises the prompt from the embeddings of real vocabulary tokens (better than random).

Three things change because the sequence got n positions longer:
  1. the padding mask key_keep (B, T) must get n True columns in front (the prompt is always "real"),
  2. the hidden states coming out have n extra positions in front: drop them before the head,
  3. positions shift by n (fine for RoPE; for Model 2 with sinusoidal positions too, they are just later positions).

How we plug it into our backbones without editing them: every backbone starts with  x = self.tok_emb(ids)
(Model 3: self.embedding(ids)). We replace that embedding module by PromptedEmbedding, which returns the
token embeddings WITH the prompt in front.

Torch basics you will need:
  emb.weight                 -> (vocab, d) embedding matrix;   emb.num_embeddings -> vocab size
  emb.weight[ids]            -> rows for these token ids
  torch.randint(0, V, (n,))  -> n random ints in [0, V)
  t[None]                    -> add a leading dim of size 1;  .expand(B, -1, -1) -> repeat it B times (no copy)
  torch.cat([a, b], dim=1)   -> join along the sequence dim
  torch.ones(B, n, dtype=torch.bool, device=...)
"""
import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    def __init__(self, tok_emb, n_virtual, init_ids=None):
        """tok_emb: the model's nn.Embedding (used only to initialise). init_ids: n_virtual token ids to copy
        the initial vectors from (random vocabulary ids if None)."""
        super().__init__()
        # Step 1. self.n = n_virtual
        # Step 2. If init_ids is None: init_ids = torch.randint(0, tok_emb.num_embeddings, (n_virtual,))
        # Step 3. self.prompt = Parameter(tok_emb.weight[init_ids].detach().clone())       -> (n, d)
        raise NotImplementedError

    def forward(self, x):
        """x: (B, T, d) token embeddings -> (B, n + T, d)."""
        # Step 1. p = self.prompt[None].expand(x.size(0), -1, -1).to(x.dtype)
        # Step 2. return torch.cat([p, x], dim=1)
        raise NotImplementedError


class PromptedEmbedding(nn.Module):
    """Drop-in replacement for the backbone's embedding: ids (B, T) -> (B, n + T, d)."""
    def __init__(self, tok_emb, soft_prompt):
        super().__init__()
        # Step 1. self.tok_emb = tok_emb (frozen elsewhere, with the rest of the model); self.soft = soft_prompt
        raise NotImplementedError

    def forward(self, ids):
        # Step 1. return self.soft(self.tok_emb(ids))
        raise NotImplementedError


def extend_keep(key_keep, n):
    """(B, T) bool padding mask -> (B, n + T), with n True columns in front. None stays None."""
    # Step 1. if key_keep is None: return None
    # Step 2. ones = torch.ones(B, n, dtype=torch.bool, device=key_keep.device); return cat([ones, key_keep], 1)
    raise NotImplementedError


def drop_prompt(h, n):
    """(B, n + T, d) hidden states -> (B, T, d): remove the prompt positions before the task head."""
    # Step 1. return h[:, n:]
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = nn.Embedding(100, 8); emb.requires_grad_(False)
    soft = SoftPrompt(emb, n_virtual=5, init_ids=torch.tensor([1, 2, 3, 4, 5]))
    print(torch.equal(soft.prompt.data, emb.weight[1:6]))              # expect True
    pe = PromptedEmbedding(emb, soft)
    ids = torch.randint(0, 100, (2, 7))
    out = pe(ids)
    print(out.shape, torch.equal(out[:, 5:], emb(ids)))                # expect torch.Size([2, 12, 8]) True
    keep = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    print(extend_keep(keep, 5).shape, extend_keep(keep, 5)[1].tolist())
    # expect torch.Size([2, 12]) [True x 9, False x 3]
    print(drop_prompt(out, 5).shape)                                   # expect torch.Size([2, 7, 8])
    out.sum().backward()
    print([n for n, p in pe.named_parameters() if p.requires_grad], soft.prompt.grad is not None)
    # expect ['soft.prompt'] True
