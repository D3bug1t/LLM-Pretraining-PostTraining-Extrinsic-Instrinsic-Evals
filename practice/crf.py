"""Linear-chain CRF over BIO tags.

Tag layout (same as the Phase 2 notebook): 0 = O, 1 + 2k = B-label_k, 2 + 2k = I-label_k.

IDEA: the linear head gives emission scores em[t, j] (how much token t looks like tag j) and each token picks its
tag ALONE, so invalid sequences like "O I-JUDGE" happen. A CRF adds learned scores for neighbouring tag pairs,
trans[i, j] = "how good is tag j right after tag i" (+ start[j] / end[j] for the first / last token), and scores a
WHOLE tag sequence y:

    score(y) = start[y0] + sum_t em[t, y_t] + sum_{t>=1} trans[y_{t-1}, y_t] + end[y_last]
    P(y) = exp(score(y)) / Z,     Z = sum over ALL possible sequences of exp(score)

Training: minimise  -log P(gold) = log Z - score(gold).
Predicting: pick the sequence with the highest score (Viterbi).
Z sums over n_tags ** T sequences, too many to list; the FORWARD ALGORITHM computes it in T steps:

    alpha[j] after token 0  = start[j] + em[0, j]
    alpha[j] after token t  = logsumexp over i of ( alpha_prev[i] + trans[i, j] )  + em[t, j]
    log Z                   = logsumexp over j of ( alpha_last[j] + end[j] )

    (alpha[j] = log of the summed exp-scores of all partial sequences ending in tag j. logsumexp instead of a plain
    sum of exps, because exp of big scores overflows; logsumexp works in log space and is exact.)
Viterbi is the same recursion with MAX instead of logsumexp, remembering which i won (back-pointers).

Torch basics you will need:
  torch.logsumexp(t, dim)          -> log(sum(exp(t))) along dim, numerically safe
  t.max(dim)                       -> (values, indices) of the max along dim
  alpha[:, None] + trans           -> broadcasting: (n,1) + (n,n) = (n,n), entry [i, j] = alpha[i] + trans[i, j]
  em[torch.arange(T), y]           -> picks em[t, y[t]] for every t at once (fancy indexing)
  trans[y[:-1], y[1:]]             -> picks trans[y[t-1], y[t]] for every consecutive pair
  self.register_buffer("name", t)  -> fixed (not trained) tensor that moves with .to(device)
  np.where(cond, a, b)             -> element-wise choice
"""
import numpy as np
import torch
import torch.nn as nn


def bio_constraints(n_tags):
    """Returns (allowed, start_ok) as numpy bool arrays:
    allowed[i, j]: tag j may directly follow tag i;  start_ok[j]: a sequence may start with tag j."""
    # Rule: an I- tag (index 2, 4, 6, ...) may only follow its own B- (index j - 1) or itself (j). Everything else is free.
    # Step 1. allowed = np.ones((n_tags, n_tags), dtype=bool); start_ok = np.ones(n_tags, dtype=bool)
    # Step 2. for each I- column j (j = 2, 4, ..., n_tags - 1): set the whole column allowed[:, j] = False,
    #         then allow the two predecessors back: allowed[j - 1, j] = allowed[j, j] = True; and start_ok[j] = False.
    # Step 3. return allowed, start_ok
    raise NotImplementedError


class CRF(nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        # Step 1. Three trainable tables, all zeros: self.trans (n, n), self.start (n,), self.end (n,) (nn.Parameter).
        # Step 2. Penalties from bio_constraints: 0.0 where allowed, -1e4 where forbidden (np.where), turned into
        #         float tensors (torch.tensor(..., dtype=torch.float32)). Register them as buffers "pen_t" (n, n)
        #         and "pen_s" (n,). -1e4 instead of -inf: big enough to never be chosen, and no NaNs in the maths.
        raise NotImplementedError

    def scores(self):
        """(trans (n, n), start (n,), end (n,)) with the BIO constraints applied."""
        # return trans + its penalty, start + its penalty, end. Always use these, never the raw parameters.
        raise NotImplementedError

    def _log_z(self, em, trans, start, end):
        """log Z of ONE sequence, em (T, n) -> scalar. The forward algorithm from the top of the file."""
        # Step 1. alpha = start + em[0]                                           -> (n,)
        # Step 2. for t in 1..T-1: alpha = logsumexp over dim 0 of (alpha[:, None] + trans), plus em[t]
        # Step 3. return logsumexp of (alpha + end) over dim 0
        raise NotImplementedError

    def _path_score(self, em, y, trans, start, end):
        """score(y) of ONE sequence: em (T, n), y (T,) long -> scalar. The formula at the top, no loop needed."""
        # start[y[0]] + (em[t, y_t] for all t, summed) + (trans[y_{t-1}, y_t] for all pairs, summed) + end[y[-1]]
        # Use torch.arange(T, device=em.device) for the indexing (the GPU needs the index on the same device).
        raise NotImplementedError

    def nll(self, em, tags):
        """em: (B, T, n) emission scores. tags: (B, T) gold tags, -100 = skip this position.
        Returns the mean negative log-likelihood per tagged token (scalar)."""
        # Step 1. trans, start, end = self.scores()
        # Step 2. For each row b of the batch:
        #           keep = tags[b] != -100               (boolean (T,))
        #           if no position is kept, skip the row
        #           e, y = em[b][keep], tags[b][keep]    (the chain simply links the kept tokens)
        #           add  _log_z(e, ...) - _path_score(e, y, ...)  to a running total; add keep.sum() to a token count
        # Step 3. return total / token count  (per-token, so it's on the same scale as cross-entropy)
        # (Later, once it's correct, I'll show you how to make _log_z fast for 1024-token windows.)
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, em):
        """em: (T, n) emission scores of one sequence. Returns the best tag path as a list of T ints."""
        # Step 1. trans, start, end = self.scores();  delta = start + em[0];  backpointers = []  (a Python list)
        # Step 2. for t in 1..T-1:
        #           cand = delta[:, None] + trans            (n, n): come from i, go to j
        #           best_val, best_i = cand.max(0)           best previous tag for every j
        #           remember best_i in backpointers; delta = best_val + em[t]
        # Step 3. last = index of the max of (delta + end)
        # Step 4. Walk backwards: path = [last]; for bp in reversed(backpointers): path.append(bp[path[-1]])
        #         then reverse path and return it as a list of ints (int(x) for each).
        raise NotImplementedError


if __name__ == "__main__":
    allowed, start_ok = bio_constraints(5)              # tags: O, B-A, I-A, B-B, I-B
    print(allowed.sum(), start_ok.tolist())             # expect 19 [True, True, False, True, False]
    crf = CRF(5)
    em = torch.randn(2, 6, 5); tags = torch.randint(0, 5, (2, 6)); tags[1, 3:] = -100
    print(crf.nll(em, tags))                            # a positive number
    print(crf.decode(em[0]))                            # 6 tags, never an I- right after a different tag / at the start
