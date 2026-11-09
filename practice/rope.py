"""Rotary position embeddings (Su et al. 2021, arXiv 2104.09864), "rotate-half" layout.

IDEA: instead of ADDING a position vector to the token embedding, RoPE ROTATES the query and key vectors
by an angle that grows with the position. Pairs of dimensions (i, i + d/2) are rotated together, each pair at
its own speed (frequency). Then q_t . k_s only depends on the DISTANCE t - s -> relative positions for free.

Rotating a 2-D point (a, b) by angle theta:   (a cos - b sin,  b cos + a sin).
Written for the whole vector at once:          x * cos + rotate_half(x) * sin
where rotate_half turns (a, b) into (-b, a). That is what the three functions below build.

How to work: delete each `raise NotImplementedError` and write the steps above it.
Torch basics you will need:
  torch.arange(start, end, step)          -> 1-D tensor [start, start+step, ...]   (add dtype=torch.float32)
  torch.outer(a, b)                       -> matrix M[i, j] = a[i] * b[j]
  torch.cat([t1, t2], dim=-1)             -> glue tensors along the last dimension
  t.chunk(2, dim=-1)                      -> split the last dimension into 2 equal halves (returns a tuple)
  t.cos(), t.sin()                        -> element-wise cos / sin
"""
import torch


def rope_cache(T, d_head, base=10000.0):
    """Returns (cos, sin), each (T, d_head), for positions 0..T-1."""
    # Step 1. Frequencies, one per PAIR of dimensions -> d_head // 2 of them:
    #         inv_freq[i] = 1 / base ** (2i / d_head)   for i = 0, 1, ..., d_head/2 - 1
    #         Hint: torch.arange(0, d_head, 2) already gives 0, 2, 4, ... = the "2i" values. Shape (d_head/2,)
    # Step 2. Angles: angle[t, i] = t * inv_freq[i]. Positions are torch.arange(T). Use torch.outer.
    #         Shape (T, d_head/2)
    # Step 3. Dimension i and dimension i + d_head/2 share an angle, so repeat the angles once:
    #         concatenate the angle matrix with itself along the last dim. Shape (T, d_head)
    # Step 4. return angles.cos(), angles.sin()
    raise NotImplementedError


def rotate_half(x):
    """x: (..., d_head) -> (..., d_head): [x1, x2] -> [-x2, x1]  (x1 = first half, x2 = second half)."""
    # Step 1. Split x into its two halves along the last dimension (chunk).
    # Step 2. Concatenate (-second half, first half) along the last dimension and return it.
    raise NotImplementedError


def apply_rope(x, cos, sin):
    """x: (B, H, T, d_head); cos/sin broadcastable to x (e.g. (1, 1, T, d_head)). Returns x rotated by position."""
    # One line: the rotation formula from the top of the file.
    # (Broadcasting: torch multiplies (1, 1, T, d) with (B, H, T, d) by repeating over the size-1 dimensions.)
    raise NotImplementedError


# Quick self-check you can run (python rope.py) once done:
if __name__ == "__main__":
    cos, sin = rope_cache(8, 4)
    print(cos.shape, sin.shape)                       # expect torch.Size([8, 4]) twice
    print(rotate_half(torch.tensor([1., 2., 3., 4.])))  # expect tensor([-3., -4., 1., 2.])
