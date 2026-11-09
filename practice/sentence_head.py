"""Sentence-level head for Phase 3 (Rhetorical Roles): one label per SENTENCE, not per token.

IDEA: we pack several consecutive sentences of a document into one window of tokens, run the backbone once,
and get a hidden vector per TOKEN, h (B, T, d). To label sentences we need one vector per SENTENCE:
mean-pool the hidden vectors of the tokens of that sentence. For that each token carries sent_idx:
        sent_idx[b, t] = index (0, 1, 2, ...) of the sentence the token belongs to inside window b,
                         or -1 for tokens that belong to no sentence (padding, separators, context-only tokens).
Packing sentences together is what makes the model CONTEXT-AWARE: thanks to bidirectional attention, every token
also sees the neighbouring sentences. Putting a single sentence per window gives the sentence-only baseline.

Then a Linear layer maps each sentence vector to 13 role scores. Training loss: cross-entropy over the sentences
that exist (present = True). Roles are very imbalanced, and the metric is MACRO-F1 (every role counts equally),
so a class-weighted loss can help: weight_c = (N / count_c) ** power, normalised so the average weight per
training sentence is 1. power = 0 -> no weighting, 1 -> full inverse frequency, 0.5 -> in between.
(A CRF over the sentence sequence, crf.py, can sit on top of these per-sentence scores: roles follow a
document order - facts, arguments, precedents, ratio, ruling.)

Torch basics you will need:
  t.clamp(min=0)                                  -> replace -1 by 0 (a valid index; the row is masked out anyway)
  h.new_zeros(B, S, d)                            -> zeros with h's dtype/device
  out.scatter_add_(1, index, src)                 -> out[b, index[b, t, k], k] += src[b, t, k]   (sums by group)
  idx[..., None].expand(-1, -1, d)                -> (B, T) -> (B, T, d) by repeating along a new last dim
  valid[..., None]                                -> (B, T) -> (B, T, 1) for broadcasting against h
"""
import torch
import torch.nn as nn


def sentence_mean_pool(h, sent_idx, n_sent):
    """h (B, T, d), sent_idx (B, T) long with -1 = ignore. Returns (pooled (B, n_sent, d), present (B, n_sent) bool):
    pooled[b, s] = mean of h[b, t] over tokens with sent_idx[b, t] == s; zeros and present=False if s has no token."""
    # Step 1. valid = sent_idx >= 0;  idx = sent_idx.clamp(min=0)
    # Step 2. sums = h.new_zeros(B, n_sent, d).scatter_add_(1, idx[..., None].expand(-1, -1, d), h * valid[..., None])
    #         (tokens that are not valid add zeros, so they do not matter)
    # Step 3. counts = h.new_zeros(B, n_sent).scatter_add_(1, idx, valid.to(h.dtype))
    # Step 4. return sums / counts.clamp(min=1)[..., None], counts > 0
    raise NotImplementedError


class SentenceHead(nn.Module):
    def __init__(self, d, n_labels, dropout=0.1):
        super().__init__()
        # Step 1. self.drop = nn.Dropout(dropout);  self.proj = nn.Linear(d, n_labels)
        raise NotImplementedError

    def forward(self, h, sent_idx, n_sent):
        """Returns (logits (B, n_sent, n_labels), present (B, n_sent))."""
        # Step 1. pooled, present = sentence_mean_pool(h, sent_idx, n_sent)
        # Step 2. return self.proj(self.drop(pooled)), present
        raise NotImplementedError


def class_weights(counts, power=0.5):
    """counts: (C,) number of training sentences per label. Returns (C,) weights w with
    w_c proportional to (N / count_c) ** power and sum_c w_c * count_c == N (average weight 1).
    Labels with count 0 get weight 0."""
    # Step 1. counts = counts.float();  N = counts.sum()
    # Step 2. w = (N / counts.clamp(min=1)) ** power;  w[counts == 0] = 0
    # Step 3. return w * N / (w * counts).sum()
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(0)
    h = torch.randn(2, 6, 4)
    sent_idx = torch.tensor([[0, 0, 1, 1, 1, -1], [0, 2, 2, -1, -1, -1]])    # item 1 has no token for sentence 1
    pooled, present = sentence_mean_pool(h, sent_idx, n_sent=3)
    print(pooled.shape, present.tolist())          # expect torch.Size([2, 3, 4]) [[True, True, False], [True, False, True]]
    print(torch.allclose(pooled[0, 1], h[0, 2:5].mean(0)), torch.allclose(pooled[1, 2], h[1, 1:3].mean(0)))
    # expect True True

    head = SentenceHead(4, 13)
    logits, present = head(h, sent_idx, 3)
    print(logits.shape)                            # expect torch.Size([2, 3, 13])

    w = class_weights(torch.tensor([90, 9, 1, 0]), power=1.0)
    print([round(x, 3) for x in w.tolist()], round((w * torch.tensor([90., 9, 1, 0])).sum().item(), 3))
    # expect [0.37, 3.704, 33.333, 0.0] 100.0   (rare labels weigh more; average weight per sentence is 1)
    print(class_weights(torch.tensor([90, 9, 1, 0]), power=0.0).tolist())   # expect [1.0, 1.0, 1.0, 0.0]
