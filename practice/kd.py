"""Knowledge distillation (Hinton et al. 2015, arXiv 1503.02531) for token classification.

IDEA: a big trained TEACHER gives, for every token, a full probability distribution over the tags, not just the
right answer ("this is 70% B-JUDGE, 25% B-A.COUNSEL..."). The small STUDENT learns from the gold tags AND from
that distribution. A temperature T > 1 softens both distributions so the small probabilities (the "dark
knowledge": which wrong tags are close) carry more signal.

    loss = (1 - alpha) * CE(student, gold)  +  alpha * T^2 * KL( softmax(teacher / T) || softmax(student / T) )

The T^2 keeps the KD gradient on the same scale as the CE gradient when T changes.

Torch basics you will need:
  F.cross_entropy(logits (N, C), targets (N,), ignore_index=-100)  -> mean CE, skipping targets == -100
  F.log_softmax(t, dim=-1), F.softmax(t, dim=-1)
  F.kl_div(student_LOG_probs, teacher_probs, reduction="batchmean")  -> KL(teacher || student) / N
       (careful: first argument is LOG-probabilities, second is plain probabilities)
  t.view(-1, C)       -> flatten (B, T, C) into (B*T, C);  tags.view(-1) -> (B*T,)
  t[mask]             -> keep only the rows where the boolean mask is True
"""
import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, tags, temperature, alpha):
    """student_logits, teacher_logits: (B, T, n_tags). tags: (B, T) gold, -100 = ignore position.
    Returns (1 - alpha) * cross-entropy(student, gold) + alpha * T^2 * KL(teacher_T || student_T),
    both averaged over the non-ignored positions only (scalar)."""
    # Step 1. Flatten: s = student_logits.view(-1, C), t = teacher_logits.view(-1, C), y = tags.view(-1)
    #         (C = student_logits.shape[-1])
    # Step 2. ce = cross-entropy of s against y with ignore_index=-100
    # Step 3. mask = y != -100;  keep only those rows of s and t (padding must not be distilled either)
    # Step 4. kl = F.kl_div(log_softmax(s_kept / T), softmax(t_kept / T), reduction="batchmean")
    # Step 5. return (1 - alpha) * ce + alpha * T * T * kl
    raise NotImplementedError


if __name__ == "__main__":
    s = torch.randn(2, 6, 5); tags = torch.randint(0, 5, (2, 6)); tags[0, :2] = -100
    print(kd_loss(s, s.clone(), tags, 2.0, 1.0))      # expect ~0: student == teacher and alpha=1 (KD term only)
    print(kd_loss(s, torch.randn(2, 6, 5), tags, 2.0, 0.0), F.cross_entropy(s.view(-1, 5), tags.view(-1)))  # equal
