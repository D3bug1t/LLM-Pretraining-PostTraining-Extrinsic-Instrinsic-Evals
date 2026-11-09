# Phase 5 post-processing: submission zips from the notebook bundles (dev / test logits), no retraining.
#   threshold: "global" (one threshold, best dev macro-F1) or "perlabel" (one per statute, fitted on dev)
#   view: "head4" (the trained input) or "both" (mean of head:4 and head:8 probabilities, where test head:8 was saved)
#   ensembles: mean probabilities of the runs in ENSEMBLES ("<model>_ens")
# CV F1 = thresholds fitted on one half of dev and scored on the other (3 random 2-fold splits), an estimate of the
# test score that the dev-fitted threshold does not inflate.
# Usage: python postprocess_phase5.py [bundle dirs...]   (default: every Phase_5/bundle_* folder)
import glob, json, os, sys, zipfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pp_work")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "phase5_pp_log.txt"), "w", encoding="utf-8")
def log(s=""):
    print(s); LOG.write(s + "\n"); LOG.flush()

GRID = np.round(np.arange(0.02, 0.92, 0.02), 2)
dev_gold = {str(r["id"]): r["labels"] for r in
            json.load(open(os.path.join(HERE, "data", "Phase_5_LSI", "Train", "dev.json"), encoding="utf-8"))}
test_ids = [str(r["id"]) for r in json.load(open(os.path.join(HERE, "data", "Phase_5_LSI", "Test", "test.json"), encoding="utf-8"))]
N = 100

def decide(P, t):                      # t: scalar or per-label vector; at least one statute per document
    Z = P >= t
    Z[np.arange(len(P)), P.argmax(1)] = True
    return Z

def macro_f1(Y, Z):
    tp = (Y & Z).sum(0); fp = (~Y & Z).sum(0); fn = (Y & ~Z).sum(0)
    d = 2 * tp + fp + fn
    return float(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0).mean())

def fit_global(P, Y):
    return float(GRID[int(np.argmax([macro_f1(Y, decide(P, g)) for g in GRID]))])

def fit_perlabel(P, Y):
    # per statute: the grid threshold with the best F1 of that statute (vectorised over statutes)
    f = []
    for g in GRID:
        Z = P >= g
        tp = (Y & Z).sum(0); fp = (~Y & Z).sum(0); fn = (Y & ~Z).sum(0)
        d = 2 * tp + fp + fn
        f.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0))
    return GRID[np.argmax(np.stack(f), 0)]

FIT = dict(global_=fit_global, perlabel=fit_perlabel)

def cv(P, Y, fit):
    rng, s = np.random.default_rng(0), []
    for _ in range(3):
        p = rng.permutation(len(P)); h = (p[:len(P) // 2], p[len(P) // 2:])
        for a, b in (h, h[::-1]):
            s.append(macro_f1(Y[b], decide(P[b], fit(P[a], Y[a]))))
    return float(np.mean(s))

def to_probs(d, ids):
    return torch.sigmoid(torch.stack([d[i].float() for i in ids])).numpy()

# ---- load runs ----
bundles = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "bundle_*")))
runs = {}
for b in bundles:
    for f in sorted(glob.glob(os.path.join(b, "*", "logits.pt"))):
        name = os.path.basename(os.path.dirname(f))
        meta = json.load(open(os.path.join(os.path.dirname(f), "val_metrics.json"), encoding="utf-8"))
        runs[name] = dict(meta=meta, logits=torch.load(f))
log(f"bundles: {bundles}")
log(f"runs: {list(runs)}")
dev_ids = sorted(set.intersection(*[set(r["logits"]["dev"]) for r in runs.values()]))
assert set(test_ids) == set.intersection(*[set(r["logits"]["test"]) for r in runs.values()]), "test ids differ"
Y = np.zeros((len(dev_ids), N), dtype=bool)
for k, i in enumerate(dev_ids):
    Y[k, dev_gold[i]] = True
log(f"dev {len(dev_ids)} documents, test {len(test_ids)} documents")

# ---- candidates: (name, model, dev probs, test probs) ----
cands = []
for name, r in runs.items():
    L, model = r["logits"], r["meta"]["model"]
    pd, pt = to_probs(L["dev"], dev_ids), to_probs(L["test"], test_ids)
    r["pd"], r["pt"] = pd, pt
    cands.append((f"{name}_head4", model, pd, pt))
    if "head:8" in L.get("dev_sections", {}) and "head:8" in L.get("test_sections", {}):
        cands.append((f"{name}_both", model, (pd + to_probs(L["dev_sections"]["head:8"], dev_ids)) / 2,
                      (pt + to_probs(L["test_sections"]["head:8"], test_ids)) / 2))
# same-model ensembles of the unweighted full-size runs (a class-weighted run's probabilities are on another scale)
ENSEMBLES = dict(m4=["m4_full", "m4_lorma"], m5=["m5_full", "m5_lorma"])
for model, names in ENSEMBLES.items():
    if all(n in runs for n in names):
        cands.append((f"{model}_ens_head4", model, np.mean([runs[n]["pd"] for n in names], 0),
                      np.mean([runs[n]["pt"] for n in names], 0)))
        log(f"{model}_ens = mean of {names}")

# ---- score, write zips ----
rows = []
for cname, model, pd, pt in cands:
    for tname, fit in FIT.items():
        t = fit(pd, Y)
        dev_f1, cv_f1 = macro_f1(Y, decide(pd, t)), cv(pd, Y, fit)
        Z = decide(pt, t)
        pred = [{"id": i, "pred": [int(j) for j in np.flatnonzero(z)]} for i, z in zip(test_ids, Z)]
        assert all(0 <= j < N for p in pred for j in p["pred"]) and all(p["pred"] for p in pred)
        tag = f"{cname}_{tname.rstrip('_')}"
        with zipfile.ZipFile(os.path.join(OUT, f"submission_{tag}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("prediction.json", json.dumps(pred))
        rows.append(dict(submission=tag, model=model, dev_f1=dev_f1, cv_f1=cv_f1,
                         dev_pred_per_doc=float(decide(pd, t).sum(1).mean()), test_pred_per_doc=float(Z.sum(1).mean())))

log()
log(f"{'submission':42s} model  dev F1   CV F1  pred/doc dev test")
for r in sorted(rows, key=lambda r: (r["model"], -r["cv_f1"])):
    log(f"{r['submission']:42s} {r['model']:5s} {r['dev_f1']:.4f}  {r['cv_f1']:.4f}   {r['dev_pred_per_doc']:5.2f} {r['test_pred_per_doc']:5.2f}")
json.dump(rows, open(os.path.join(OUT, "phase5_pp_summary.json"), "w"), indent=1)
log(f"\nzips in {OUT}")
