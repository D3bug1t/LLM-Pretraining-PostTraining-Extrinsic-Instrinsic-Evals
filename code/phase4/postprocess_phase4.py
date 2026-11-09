"""Phase 4 post-processing (local, CPU): decision rules and ensembles from the saved logits of every run.

Run from this folder:  python postprocess_phase4.py
Reads phase4_results_bundle*/<run>/logits.pt, writes pp_work/submission_{base}_{strategy}.zip and pp_work/phase4_pp_log.txt.

Views (which part of the case was read):  tail = the run's own input (last 512 tokens);  tail8 = the last 8 x 512 tokens
(the `tail:8` test_sections logits);  both = the average of the two views' probabilities.
Bases: every run x view, and probability-averaged ensembles of the runs of one model (m4_ens, m5_ens, ...) and all runs.
The KD student is in no ensemble.
Strategies (score s = log p1 - log p0, predict 1 if s > t):
  argmax  t = 0
  prior   t = log(n1 / n0) of the TRAINING labels: removes the training class prior (41% label 1), i.e. assumes balanced
          classes, as in dev
  devthr  t = the threshold that maximizes dev macro-F1 (fitted on dev, so its dev score is optimistic)
"""
import os, glob, json, zipfile, math
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pp_work"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(HERE, "data", "Phase_4_CJPE")
_log = open(os.path.join(OUT, "phase4_pp_log.txt"), "w", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); _log.write(s + "\n")

# ---- labels: dev cases that are not test cases (as in the notebook); training class counts ----
test_ids = [x["id"] for x in json.load(open(os.path.join(DATA, "Test", "test.json"), encoding="utf-8"))]   # ids only

runs = {}
for f in sorted(glob.glob(os.path.join(HERE, "phase4_results_bundle*", "*", "logits.pt"))):
    name = os.path.basename(os.path.dirname(f))
    vm = json.load(open(os.path.join(os.path.dirname(f), "val_metrics.json")))
    runs[name] = dict(logits=torch.load(f), vm=vm, val_pred={x["id"]: x["pred"] for x in
                      json.load(open(os.path.join(os.path.dirname(f), "val_predictions.json")))})
assert runs, "no phase4_results_bundle*/<run>/logits.pt found"
dev_ids = sorted(next(iter(runs.values()))["logits"]["dev"])
dev_raw = json.load(open(os.path.join(DATA, "Train", "dev.json"), encoding="utf-8"))
_tset = set(test_ids)
gold = {}
for x in dev_raw:
    if x["id"] in _tset: continue
    gold.setdefault(x["id"], int(x["label"]))
assert set(dev_ids) <= set(gold), "dev ids of the runs not found in dev.json"
y = np.array([gold[d] for d in dev_ids])
tr = json.load(open(os.path.join(DATA, "Train", "train.json"), encoding="utf-8"))
_seen = {}
for x in tr: _seen.setdefault(x["text"], int(x["label"]))
n1 = sum(_seen.values()); n0 = len(_seen) - n1
PRIOR_T = math.log(n1 / n0)
del tr, _seen, dev_raw
log(f"runs: {sorted(runs)}")
log(f"dev cases {len(dev_ids)} (label 1: {y.mean():.3f}); training unique cases 0={n0} 1={n1} -> prior threshold {PRIOR_T:+.4f}")

# ---- per-run views: log-probabilities (n, 2) on dev and test ----
def lp(d, ids):
    return torch.log_softmax(torch.stack([d[i].float() for i in ids]), -1).numpy()

views = {}          # (run, view) -> (dev logp, test logp)
for name, r in runs.items():
    L = r["logits"]
    views[(name, "tail")] = (lp(L["dev"], dev_ids), lp(L["test"], test_ids))
    if "tail:8" in L.get("dev_sections", {}) and "tail:8" in L.get("test_sections", {}):
        views[(name, "tail8")] = (lp(L["dev_sections"]["tail:8"], dev_ids), lp(L["test_sections"]["tail:8"], test_ids))
        views[(name, "both")] = tuple(np.log((np.exp(a) + np.exp(b)) / 2) for a, b in zip(views[(name, "tail")], views[(name, "tail8")]))
    # check: the notebook's own dev predictions = argmax of the tail view
    assert all(int(v.argmax()) == r["val_pred"][d] for d, v in zip(dev_ids, views[(name, "tail")][0])), name

def ensemble(members, view):
    ms = [views[(m, view)] for m in members if (m, view) in views]
    if len(ms) < len(members): return None
    return tuple(np.log(np.mean([np.exp(m[k]) for m in ms], 0)) for k in (0, 1))

bases = {f"{n}_{v}" if v != "tail" else n: views[(n, v)] for n, v in views}
# ensembles: only the standard input (tail, 1 chunk); the KD student and the section / hierarchy experiments stay out.
# Runs on all training cases (max_train_docs None) form their own group.
by_model, alldocs = defaultdict(list), defaultdict(list)
for n, r in runs.items():
    c = r["vm"]["config"]
    if r["vm"]["method"] == "kd" or c["section"] != "tail" or c["n_chunks"] != 1: continue
    (by_model if c["max_train_docs"] else alldocs)[r["vm"]["model"]].append(n)
groups = {f"{m}_ens": ns for m, ns in sorted(by_model.items()) if len(ns) > 1}
groups.update({f"{m}_alldocs_ens": ns for m, ns in sorted(alldocs.items()) if len(ns) > 1})
for m in sorted(alldocs):
    if by_model.get(m) and len(alldocs[m]) > 1: groups[f"{m}_both_sets_ens"] = by_model[m] + alldocs[m]
if len(by_model) > 1: groups["all_ens"] = [n for m, ns in by_model.items() if m != "m3" for n in ns]   # m3 (GRU) left out, as in Phase 3
for g, ns in groups.items():
    log(f"ensemble {g}: {ns}")
    for v in ("tail", "tail8", "both"):
        e = ensemble(ns, v)
        if e is not None: bases[g if v == "tail" else f"{g}_{v}"] = e

# ---- strategies ----
def metrics(pred):
    return (f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0), precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))

def best_threshold(s, idx=None):
    idx = np.arange(len(s)) if idx is None else idx
    ss, yy = s[idx], y[idx]
    o = np.argsort(-ss); ss, yy = ss[o], yy[o]           # predict 1 for the top k scores, k = 0..n
    P, N = yy.sum(), len(yy) - yy.sum()
    tp = np.concatenate([[0], np.cumsum(yy)]); fp = np.concatenate([[0], np.cumsum(1 - yy)])
    fn, tn = P - tp, N - fp
    f1 = lambda a, b, c: np.where(2 * a + b + c > 0, 2 * a / np.maximum(1, 2 * a + b + c), 0.0)
    k = int(np.argmax((f1(tp, fp, fn) + f1(tn, fn, fp)) / 2))
    if k == 0: return float(ss[0] + 1)
    if k == len(ss): return float(ss[-1] - 1)
    return float((ss[k - 1] + ss[k]) / 2)

def cv_threshold_f1(s, reps=20):
    """Honest estimate for devthr: fit the threshold on a random half of dev, score the other half (both ways)."""
    rng, out = np.random.RandomState(0), []
    for _ in range(reps):
        perm = rng.permutation(len(s)); a, b = perm[:len(s) // 2], perm[len(s) // 2:]
        for fit, ev in ((a, b), (b, a)):
            t = best_threshold(s, fit)
            out.append(f1_score(y[ev], (s[ev] > t).astype(int), average="macro", labels=[0, 1]))
    return float(np.mean(out))

rows = []
for b, (dv, te) in bases.items():
    sd, st = dv[:, 1] - dv[:, 0], te[:, 1] - te[:, 0]
    for strat, t in (("argmax", 0.0), ("prior", PRIOR_T), ("devthr", best_threshold(sd))):
        pd_, pt = (sd > t).astype(int), (st > t).astype(int)
        f, p, r = metrics(pd_)
        rows.append((b, strat, t, f, p, r, pt.mean(), cv_threshold_f1(sd) if strat == "devthr" else None))
        fn = os.path.join(OUT, f"submission_{b}_{strat}.zip")
        with zipfile.ZipFile(fn, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("prediction.json", json.dumps([{"id": i, "pred": int(v)} for i, v in zip(test_ids, pt)]))

log(f"\n{'base':22s} {'strategy':8s} {'thresh':>7s} {'dev F1':>7s} {'P':>6s} {'R':>6s} {'test %1':>7s} {'CV F1':>6s}")
for b, s, t, f, p, r, tp, cv in rows:
    log(f"{b:22s} {s:8s} {t:+7.3f} {f:7.4f} {p:6.4f} {r:6.4f} {100 * tp:6.1f}%" + (f" {cv:6.4f}" if cv is not None else ""))
log(f"\n{len(rows)} zips in {OUT}")
log("devthr: its dev F1 is fitted on dev (optimistic); CV F1 = threshold fitted on one half of dev, scored on the other "
    "half (20 random splits x 2), comparable with the argmax / prior dev F1")
