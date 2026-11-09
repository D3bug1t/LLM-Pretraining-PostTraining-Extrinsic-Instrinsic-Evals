"""Phase 3 post-processing: decoding strategies and ensembles from the saved per-sentence logits (no retraining).

Reads every <run>/logits.pt (dev + test logits per document, saved by phase3_finetune_kaggle.ipynb) under RESULTS_GLOB
and writes pp_work/submission_{base}_{strategy}.zip, where base = a run or an ensemble of runs, strategy =
  argmax         highest-probability role per sentence (= the run's own submission)
  prior          logit adjustment: log p(role | sentence) - TAU * log p(role), p(role) = training frequency
                 (macro-F1 counts rare roles as much as frequent ones; plain argmax favours the frequent ones)
  viterbi        best role sequence per document: sum of log p(role | sentence) + log p(role_t | role_t-1),
                 transition probabilities counted on the training documents
  prior_viterbi  both: the hybrid NN/HMM decode (posterior / prior ~ likelihood, combined with the transitions)
Ensembles average the runs' probabilities. TAU, the transition counts and the ensembles are fixed in advance from
the training data / run list, not tuned on dev. Dev scores are printed for every candidate (same scoring as
evaluate_phase3.py) and written to pp_work/phase3_pp_log.txt.

Run from Phase_3/:  python postprocess_phase3.py
"""
import glob, json, os, zipfile
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

RESULTS_GLOB = "phase3_results_bundle*/*/logits.pt"
DATA = "data/Phase_3_RR"
SKIP_RUNS = []                      # run names to leave out entirely
ENSEMBLES = {                       # name -> runs (probabilities averaged); runs not found are skipped
    "m2_ens": ["m2_lora", "m2_rslora", "m2_lorma"],
    "m4_ens": ["m4_lora", "m4_rslora", "m4_lorma", "m4_full"],
    "m5_ens": ["m5_lora", "m5_rslora", "m5_lorma", "m5_full"],
    "m45_ens": [f"{p}_{m}" for p in ("m4", "m5") for m in ("lora", "rslora", "lorma", "full")],
    "all_ens": [m for p in ("m2", "m4", "m5") for m in (f"{p}_lora", f"{p}_rslora", f"{p}_lorma", f"{p}_full")],  # m3 left out
    "m5_crf_ens": ["m5_lora_crf", "m5_rslora_crf", "m5_lorma_crf", "m5_full_crf"],
    "m5_all_ens": [f"m5_{m}{h}" for h in ("", "_crf") for m in ("lora", "rslora", "lorma", "full")],
    "m5_crf_all_ens": ["m5_lora_crf_all", "m5_rslora_crf_all", "m5_lorma_crf_all", "m5_full_crf_all"],   # trained on dev
    "m5_crf_ens8": [f"m5_{m}_crf{s}" for s in ("", "_s1") for m in ("lora", "rslora", "lorma", "full")],  # 2 seeds
}                                   # (m3 has a single run: it gets the strategies on its own, no ensemble)
# All linear-head runs: probabilities averaged. All CRF runs: emissions and learned transition scores averaged, then
# Viterbi. A mix: every CRF run is turned into per-sentence posterior probabilities (forward-backward with its learned
# transitions) and averaged with the linear runs' softmax probabilities; the result is decoded like a linear run.
TAU = 1.0                           # prior: strength of the logit adjustment
STRATEGIES = ["argmax", "prior", "viterbi", "prior_viterbi",      # linear-head runs and their ensembles
              "dprior_dviterbi",                                  # prior + transitions counted on the document's DOMAIN
              "prior_pos_viterbi",                                # + position ratio p(role | position) / p(role)
              "dprior_pos_dviterbi",                              # both: domain prior, domain transitions, domain position
              "ef1", "hmm_ef1", "hmm_pos_ef1"]                    # expected macro-F1 decoding (see ef1_decode)
POS_BINS = 10                       # relative position in the document (sentence t of n -> bin floor(POS_BINS * t / n))
CRF_STRATEGIES = ["crf"]            # CRF runs: Viterbi with the LEARNED transitions (= the run's own submission), plus
                                    # STRATEGIES applied to the CRF's forward-backward posteriors (argmax of the posteriors =
                                    # posterior decoding; viterbi / prior_viterbi add the training-count transitions on top)
OUT = "pp_work"

os.makedirs(OUT, exist_ok=True)
_log = open(os.path.join(OUT, "phase3_pp_log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _log.write(s + "\n")

# ---- data ----
vocab = json.load(open(f"{DATA}/Train/label_vocab.json", encoding="utf-8"))
LABELS = sorted(vocab, key=vocab.get); L2I = {l: i for i, l in enumerate(LABELS)}; C = len(LABELS)
GOLD, DOMAIN, TRAIN_SEQ = {}, {}, []
for dom in ("CL", "IT"):
    for x in json.load(open(f"{DATA}/Train/{dom}_train.json", encoding="utf-8")):
        TRAIN_SEQ.append((dom, [L2I[l] for l in x["labels"]]))
    for x in json.load(open(f"{DATA}/Train/{dom}_dev.json", encoding="utf-8")):
        GOLD[x["id"]] = x["labels"]; DOMAIN[x["id"]] = dom
    for d, sents in json.load(open(f"{DATA}/Test/{dom}_test.json", encoding="utf-8")).items():
        DOMAIN[d] = dom
DEV_IDS = sorted(GOLD)

# training statistics (add-one smoothing): role prior, start / transition / end probabilities, and the role distribution
# per relative-position bin; for all training documents ("all") and per domain ("CL", "IT")
def train_stats(seqs):
    cnt = Counter(l for s in seqs for l in s)
    prior = np.log(np.array([cnt[k] + 1 for k in range(C)], float) / (sum(cnt.values()) + C))
    tr = np.ones((C, C)); st = np.ones(C); en = np.ones(C); pos = np.ones((POS_BINS, C))
    for s in seqs:
        st[s[0]] += 1; en[s[-1]] += 1
        for a, b in zip(s, s[1:]): tr[a, b] += 1
        for t, l in enumerate(s): pos[POS_BINS * t // len(s), l] += 1
    out_total = tr.sum(1) + en                  # every role is followed by a role or by the end of the document
    return dict(prior=prior, trans=np.log(tr / out_total[:, None]), end=np.log(en / out_total),
                start=np.log(st / st.sum()), pos=np.log(pos / pos.sum(1, keepdims=True)))
STATS = {"all": train_stats([s for _, s in TRAIN_SEQ])}
STATS.update({dom: train_stats([s for d, s in TRAIN_SEQ if d == dom]) for dom in ("CL", "IT")})
LOG_PRIOR, LOG_TRANS, LOG_START, LOG_END = (STATS["all"][k] for k in ("prior", "trans", "start", "end"))

def score(pred, ids):
    y = [l for d in ids for l in GOLD[d]]; p = [l for d in ids for l in pred[d]]
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=LABELS, average=None, zero_division=0)
    return float(F.mean()), float(P.mean()), float(R.mean())

def viterbi(em, trans=LOG_TRANS, start=LOG_START, end=LOG_END):
    n = len(em); bp = np.zeros((n, C), np.int64); sc = start + em[0]
    for t in range(1, n):
        m = sc[:, None] + trans; bp[t] = m.argmax(0); sc = m.max(0) + em[t]
    path = np.zeros(n, np.int64); path[-1] = (sc + end).argmax()
    for t in range(n - 1, 0, -1): path[t - 1] = bp[t, path[t]]
    return path

def _lse(x, axis):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)

def crf_marginals(em, trans, start, end):
    """log p(role of sentence t | document) under a CRF: forward-backward, (n, C)."""
    n = len(em); a = np.zeros((n, C)); b = np.zeros((n, C))
    a[0] = start + em[0]
    for t in range(1, n): a[t] = _lse(a[t - 1][:, None] + trans, 0) + em[t]
    b[n - 1] = end
    for t in range(n - 2, -1, -1): b[t] = _lse(trans + (em[t + 1] + b[t + 1])[None, :], 1)
    return a + b - _lse(a[n - 1] + end, 0)

def decode(lp, strategy, crf=None, dom=None):
    """lp: (n, C) log-probabilities (CRF runs: emissions) of one document -> list of labels.
    Strategies starting with "d" use the statistics of the document's domain, the others those of all training docs."""
    if strategy == "crf":
        return [LABELS[i] for i in viterbi(lp, *crf)]
    S = STATS[dom if strategy.startswith("d") else "all"]
    em = lp - TAU * S["prior"] if "prior" in strategy else lp
    if "pos" in strategy:
        bins = (POS_BINS * np.arange(len(lp))) // len(lp)
        em = em + S["pos"][bins] - S["prior"]
    k = viterbi(em, S["trans"], S["start"], S["end"]) if "viterbi" in strategy else em.argmax(-1)
    return [LABELS[i] for i in k]

def doc_posterior(lp, strategy):
    """(n, C) role probabilities of one document. ef1: the model's posteriors. hmm_*: forward-backward over the
    prior-adjusted scores (+ position ratio) with the training transitions (the probabilistic prior_viterbi)."""
    if not strategy.startswith("hmm"):
        return np.exp(lp - _lse(lp, 1)[:, None])
    S = STATS["all"]; em = lp - TAU * S["prior"]
    if "pos" in strategy:
        em = em + S["pos"][(POS_BINS * np.arange(len(lp))) // len(lp)] - S["prior"]
    return np.exp(crf_marginals(em, S["trans"], S["start"], S["end"]))

def ef1_decode(post):
    """Expected macro-F1 decoding over a whole set of documents (the metric is corpus-level): with the role
    probabilities P as soft counts, TP_k = sum of P[i, k] over the sentences labelled k and n_true_k = sum_i P[i, k];
    coordinate ascent from the argmax on sum_k 2 TP_k / (n_pred_k + n_true_k). Uses no labels and no parameters.
    post: doc id -> (n, C) probabilities. Returns doc id -> list of labels."""
    ids = list(post); P = np.concatenate([post[d] for d in ids]); n = len(P)
    ntrue = P.sum(0); y = P.argmax(1)
    tp = np.bincount(y, weights=P[np.arange(n), y], minlength=C); npred = np.bincount(y, minlength=C).astype(float)
    for _ in range(50):
        changed = 0
        for i in range(n):
            a = y[i]
            old = 2 * tp / (npred + ntrue)
            gain = 2 * (tp + P[i]) / (npred + 1 + ntrue) - old                    # adding sentence i to role b
            gain += 2 * (tp[a] - P[i, a]) / (npred[a] - 1 + ntrue[a]) - old[a]    # removing it from role a
            gain[a] = 0.0
            b = int(gain.argmax())
            if gain[b] > 1e-12:
                tp[a] -= P[i, a]; npred[a] -= 1; tp[b] += P[i, b]; npred[b] += 1; y[i] = b; changed += 1
        if not changed: break
    out, k = {}, 0
    for d in ids:
        out[d] = [LABELS[j] for j in y[k:k + len(post[d])]]; k += len(post[d])
    return out

def decode_set(lps, strategy, crf, ids):
    if strategy.endswith("ef1"):
        return ef1_decode({d: doc_posterior(lps[d], strategy) for d in ids})
    return {d: decode(lps[d], strategy, crf, DOMAIN[d]) for d in ids}

# ---- runs ----
RUNS = {}
for f in sorted(glob.glob(RESULTS_GLOB)):
    name = os.path.basename(os.path.dirname(f))
    if name in SKIP_RUNS: continue
    if name in RUNS: log(f"warning: run {name} found twice, keeping {RUNS[name]['path']}"); continue
    z = torch.load(f, map_location="cpu", weights_only=False)
    crf = z.get("decoder")                       # (trans, start, end) of a CRF head, None for the linear head
    conv = (lambda v: v.float().numpy()) if crf is not None else (lambda v: torch.log_softmax(v.float(), -1).numpy())
    vm = os.path.join(os.path.dirname(f), "val_metrics.json")
    tov = os.path.exists(vm) and bool(json.load(open(vm))["config"].get("train_on_val"))
    RUNS[name] = dict(path=f, tov=tov, crf=None if crf is None else tuple(np.asarray(x, float) for x in crf),
                      **{s: {d: conv(v) for d, v in z[s].items()} for s in ("dev", "test")})
assert RUNS, f"no logits.pt found under {RESULTS_GLOB}"
TEST_IDS = sorted(next(iter(RUNS.values()))["test"])
log(f"runs: {sorted(RUNS)}")
if any(r["tov"] for r in RUNS.values()):
    log(f"trained on the dev documents (dev scores meaningless, shown as n/a): {sorted(n for n, r in RUNS.items() if r['tov'])}")
log(f"training prior: " + ", ".join(f"{l}={np.exp(p):.3f}" for l, p in zip(LABELS, LOG_PRIOR)))

BASES = {n: r for n, r in RUNS.items()}
for en_name, members in ENSEMBLES.items():
    members = [m for m in (members or sorted(RUNS)) if m in RUNS]
    if len(members) < 2: continue
    kinds = {RUNS[m]["crf"] is not None for m in members}
    tov = any(RUNS[m]["tov"] for m in members)
    if len(kinds) > 1:
        lp = lambda m, sp, d: crf_marginals(RUNS[m][sp][d], *RUNS[m]["crf"]) if RUNS[m]["crf"] is not None else RUNS[m][sp][d]
        BASES[en_name] = dict(crf=None, tov=tov, **{sp: {d: np.log(np.mean([np.exp(lp(m, sp, d)) for m in members], 0))
                                                for d in RUNS[members[0]][sp]} for sp in ("dev", "test")})
        log(f"ensemble {en_name}: {members} (CRF runs as forward-backward posteriors, averaged with the linear runs)")
        continue
    if kinds == {True}:
        BASES[en_name] = dict(crf=tuple(np.mean([RUNS[m]["crf"][i] for m in members], 0) for i in range(3)), tov=tov,
                              **{s: {d: np.mean([RUNS[m][s][d] for m in members], 0) for d in RUNS[members[0]][s]}
                                 for s in ("dev", "test")})
    else:
        BASES[en_name] = dict(crf=None, tov=tov, **{s: {d: np.log(np.mean([np.exp(RUNS[m][s][d]) for m in members], 0))
                                               for d in RUNS[members[0]][s]} for s in ("dev", "test")})
    log(f"ensemble {en_name}: {members}" + (" (CRF: emissions + transitions averaged)" if kinds == {True} else ""))

# ---- candidates ----
rows = []
for base, r in BASES.items():
    if r["crf"] is not None:                         # posteriors once per base, for the non-"crf" strategies
        r = dict(r, **{f"{sp}_post": {d: crf_marginals(em, *r["crf"]) for d, em in r[sp].items()} for sp in ("dev", "test")})
    for strat in (CRF_STRATEGIES + STRATEGIES if r["crf"] is not None else STRATEGIES):
        src = (lambda sp: r[sp]) if (r["crf"] is None or strat == "crf") else (lambda sp: r[f"{sp}_post"])
        dev = decode_set(src("dev"), strat, r["crf"], DEV_IDS)
        f1, p, rc = score(dev, DEV_IDS)
        cl = score(dev, [d for d in DEV_IDS if DOMAIN[d] == "CL"])[0]; it = score(dev, [d for d in DEV_IDS if DOMAIN[d] == "IT"])[0]
        test = decode_set(src("test"), strat, r["crf"], TEST_IDS)
        assert all(len(test[d]) == len(r["test"][d]) for d in TEST_IDS)
        name = f"submission_{base}_{strat}"
        pj = os.path.join(OUT, name + ".json")
        json.dump(test, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
        with zipfile.ZipFile(os.path.join(OUT, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pj, "prediction.json")
        os.remove(pj)
        if r["tov"]: f1 = p = rc = cl = it = float("nan")
        rows.append((base, strat, f1, p, rc, cl, it))

log(f"\n{'base':14s} {'strategy':20s} {'dev F1':>7s} {'P':>6s} {'R':>6s} {'CL F1':>6s} {'IT F1':>6s}   zip")
for base, strat, f1, p, rc, cl, it in rows:
    nums = "    n/a (trained on dev)          " if f1 != f1 else f"{f1:7.4f} {p:6.4f} {rc:6.4f} {cl:6.4f} {it:6.4f}"
    log(f"{base:14s} {strat:20s} {nums}   submission_{base}_{strat}.zip")
log(f"\n{len(rows)} submission zips in {os.path.abspath(OUT)}")
_log.close()
