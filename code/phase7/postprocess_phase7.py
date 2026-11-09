"""Phase 7 post-processing: fuse each run's dense scores with BM25, normalise across queries, build ensembles.

Inputs : data/Phase_7_PCR (official files), results/<run>/emb.pt (dev + test case embeddings from the notebook).
Outputs: pp_work/submission_<family>.zip (prediction.json at the root), pp_work/pp_report.txt, pp_work/pp_results.json.

Score of candidate c for query q (only candidates allowed by the rules: not the query itself, not decided after it):
    F      = rowz( w_bm25 * rowz(BM25) + mean_i rowz(cos_i) )      rowz = z-score over the query's allowed candidates
    score  = F + lam * colz(F) - pen * [c is another query of the split]
colz = z-score of a candidate's column over all queries of the split (unsupervised: a case is relevant to few queries).
(w_bm25, lam, pen) are picked on dev for every family; the honest estimate is the 2-fold cross-validated dev F1@7
(pick on one half of the dev queries, score on the other half; 20 random splits).
"""
import glob, json, os, re, sys, zipfile, warnings
import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "Phase_7_PCR")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "pp_work")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from evaluate_phase7 import validate                                          # the official format checks
warnings.filterwarnings("ignore", category=RuntimeWarning)

GRID_BM25 = [0, .25, .5, 1, 1.5, 2, 3, 5, 8]
GRID_LAM = [0, .5, 1, 2, 3, 5, 8, 15, 30]
GRID_PEN = [0, 1, 2, 4, 8]
CV_REPEATS, CV_SEED = 20, 0

REPORT = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); REPORT.append(s)

# ---- data (same rules as the notebook) ----
CITATOR = re.compile(r"CITATOR INFO\s*:.*?(?=\bACT:|\bHEADNOTE:|$)", re.S)
DATE_RE = re.compile(r"DATE OF JUDGMENT\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{4})")
YEAR_RE = re.compile(r"\b(18[5-9]\d|19\d\d|20[0-2]\d)\b")

def clean_text(sents):
    head = CITATOR.sub(" ", " ".join(sents[:3]))
    return " ".join([head] + sents[3:])

def case_dates(text):
    m = DATE_RE.search(text[:3000])
    if m:
        y = int(m[3]) + (int(m[2]) - 1) / 12 + (int(m[1]) - 1) / 372
        return y, y
    ys = [int(y) for y in YEAR_RE.findall(text)]
    return (max(ys) if ys else None), None

DOCS, SPLITS = {}, {}
for split, folder in (("dev", "Dev"), ("test", "Test")):
    rd = lambda n: json.load(open(os.path.join(DATA, folder, n), encoding="utf-8"))
    q, c = rd(f"{split}.json"), rd(f"{split}_candidates.json")
    for d, x in list(c.items()) + list(q.items()):
        if d not in DOCS:
            t = clean_text(x["text"]); lo, hi = case_dates(t); DOCS[d] = dict(text=t, lo=lo, hi=hi)
    rel = None if split == "test" else {k: [y for y in v["relevant_candidates"] if y.strip()] for k, v in q.items()}
    queries = sorted(q) if rel is None else sorted(k for k in q if rel[k])
    pool = sorted(c)
    S = dict(queries=queries, pool=pool, rel=rel, all_queries=sorted(q))
    pi = {d: j for j, d in enumerate(pool)}
    M = np.ones((len(queries), len(pool)), bool)
    lo = np.array([np.nan if DOCS[d]["lo"] is None else DOCS[d]["lo"] for d in pool])
    for i, qq in enumerate(queries):
        if qq in pi: M[i, pi[qq]] = False
        if DOCS[qq]["hi"] is not None: M[i, lo > DOCS[qq]["hi"]] = False
    S["mask"] = M
    S["isq"] = np.array([d in q for d in pool], np.float32)[None, :]        # pool case that is a query of this split
    if rel is not None:
        G = np.zeros_like(M)
        for i, qq in enumerate(queries):
            for y in rel[qq]: G[i, pi[y]] = True
        S["gold"] = G; S["n_rel"] = G.sum(1)
    SPLITS[split] = S

def bm25(split, k1=1.2, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sp
    S = SPLITS[split]
    cv = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=2, dtype=np.float32)
    D = cv.fit_transform([DOCS[d]["text"] for d in S["pool"]]).tocsr()
    Q = cv.transform([DOCS[d]["text"] for d in S["queries"]]).tocsr()
    N, df = D.shape[0], np.bincount(D.indices, minlength=D.shape[1])
    idf = np.log(1 + (N - df + .5) / (df + .5)).astype(np.float32)
    dl = np.asarray(D.sum(1)).ravel(); C = D.tocoo()
    W = sp.csr_matrix((C.data * (k1 + 1) / (C.data + k1 * (1 - b + b * dl[C.row] / dl.mean())) * idf[C.col], (C.row, C.col)), D.shape)
    Q.data = np.log1p(Q.data)
    return (Q @ W.T).toarray().astype(np.float32)

def rowz(X, M):
    X = np.where(M, X, np.nan)
    return (X - np.nanmean(X, 1, keepdims=True)) / (np.nanstd(X, 1, keepdims=True) + 1e-6)

def colz(Z):
    mu, sd = np.nanmean(Z, 0, keepdims=True), np.nanstd(Z, 0, keepdims=True)
    mu = np.where(np.isfinite(mu), mu, 0); sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1)
    return (Z - mu) / sd

# ---- score sources ----
Z = {s: {} for s in SPLITS}                       # split -> source name -> row-z scores (nan = not allowed)
for s in SPLITS:
    Z[s]["bm25"] = rowz(bm25(s), SPLITS[s]["mask"])
RUNS = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RES, "*", "emb.pt")))
MAIN = {}
for r in RUNS:
    emb = torch.load(os.path.join(RES, r, "emb.pt"), map_location="cpu", weights_only=False)
    MAIN[r] = f"{r}|{next(iter(emb))}"                                   # the training section comes first
    for sec, sp_ in emb.items():
        for s in SPLITS:
            ix = {d: i for i, d in enumerate(sp_[s]["ids"])}; E = sp_[s]["emb"].float().numpy()
            S = SPLITS[s]
            Z[s][f"{r}|{sec}"] = rowz(E[[ix[q] for q in S["queries"]]] @ E[[ix[d] for d in S["pool"]]].T, S["mask"])
log("runs:", ", ".join(RUNS))
log("dense sources:", ", ".join(k for k in Z["dev"] if k != "bm25"))

def model_of(r): return r.split("_")[0]
FAMILIES = {}                                     # family -> list of dense sources (empty = BM25 only)
FAMILIES["bm25"] = []
for k in Z["dev"]:
    if k != "bm25":
        FAMILIES[k.split("|")[0] if k in MAIN.values() else k.replace("|", "__").replace(":", "")] = [k]
for m in sorted({model_of(r) for r in RUNS}):
    rs = [MAIN[r] for r in RUNS if model_of(r) == m]
    if len(rs) > 1: FAMILIES[f"{m}_ens"] = rs
views = [k for k in Z["dev"] if k.startswith("m5_lorma|")]
if len(views) > 1:
    FAMILIES["m5_lorma_views"] = views
    FAMILIES["m5_ens_views"] = views + [MAIN[r] for r in RUNS if model_of(r) == "m5" and r != "m5_lorma"]
FAMILIES["all_ens"] = [MAIN[r] for r in RUNS]

def score(split, srcs, wb, lam, pen):
    S = SPLITS[split]
    F = wb * Z[split]["bm25"] if srcs == [] or wb else 0
    if srcs: F = F + sum(Z[split][k] for k in srcs) / len(srcs)
    F = rowz(F, S["mask"])
    if lam: F = F + lam * colz(F)
    if pen: F = F - pen * S["isq"]
    return np.where(np.isnan(F), -np.inf, F)

def top7(X):
    part = np.argpartition(-X, 7, axis=1)[:, :7]
    return np.take_along_axis(part, np.argsort(-np.take_along_axis(X, part, 1), 1), 1)

def per_query_tp(X):
    return np.take_along_axis(SPLITS["dev"]["gold"], top7(X), 1).sum(1)

def micro_f1(tp, idx=None):
    nr = SPLITS["dev"]["n_rel"]
    if idx is not None: tp, nr = tp[idx], nr[idx]
    t = tp.sum(); return 2 * t / (7 * len(tp) + nr.sum())               # 2tp / (2tp + fp + fn)

# ---- grid on dev, 2-fold CV ----
nq = len(SPLITS["dev"]["queries"])
rng = np.random.default_rng(CV_SEED)
FOLDS = [rng.permutation(nq) for _ in range(CV_REPEATS)]
results = {}
for fam, srcs in FAMILIES.items():
    grid = [(wb, lam, pen) for wb in (GRID_BM25 if srcs else [1]) for lam in GRID_LAM for pen in GRID_PEN]
    TP = np.stack([per_query_tp(score("dev", srcs, *g)) for g in grid])       # (configs, queries)
    raw = micro_f1(per_query_tp(score("dev", srcs, 0 if srcs else 1, 0, 0)))
    # the penalty bets that test is like dev (2% of relevant cases are other queries) and not like train (19%):
    # every family is also selected without it ("<family>__nopen")
    for suffix, keep in (("", lambda g: True), ("__nopen", lambda g: g[2] == 0)):
        ix = [i for i, g in enumerate(grid) if keep(g)]
        best = max(ix, key=lambda i: micro_f1(TP[i]))
        cv = []
        for perm in FOLDS:
            a, b = perm[: nq // 2], perm[nq // 2:]
            tp = np.zeros(nq)
            for fit, held in ((a, b), (b, a)):
                g = max(ix, key=lambda i: micro_f1(TP[i], fit)); tp[held] = TP[g, held]
            cv.append(micro_f1(tp))
        r = results[fam + suffix] = dict(sources=srcs, raw_dev=raw, config=dict(zip(("w_bm25", "lam", "pen"), grid[best])),
                                         dev=micro_f1(TP[best]), cv_mean=float(np.mean(cv)), cv_sd=float(np.std(cv)))
        # sensitivity: best dev F1 with BM25 / the column term turned off
        for name, off in (("no_bm25", lambda g: g[0] == 0), ("no_col", lambda g: g[1] == 0)):
            if srcs or name != "no_bm25":
                r[name] = max(micro_f1(TP[i]) for i in ix if off(grid[i]))

# ---- report ----
log("\nDev Micro-F1@7 (118 queries). raw = the run's own cosine ranking with the date/self rules;"
    "\ndev = best (w_bm25, lam, pen) picked on all of dev (optimistic); cv = picked on half, scored on the other half."
    "\nno_* = best dev F1 with that step turned off.")
for title, pick in (("with the query-in-pool penalty", lambda f: not f.endswith("__nopen")),
                    ("without it (__nopen)", lambda f: f.endswith("__nopen"))):
    log(f"\n{title}")
    log(f"{'family':30s} {'raw':>6s} {'dev':>6s} {'cv':>13s}  {'w_bm25':>6s} {'lam':>4s} {'pen':>4s}  {'no_bm25':>7s} {'no_col':>6s}")
    for fam, r in sorted(((f, r) for f, r in results.items() if pick(f)), key=lambda kv: -kv[1]["cv_mean"]):
        c = r["config"]
        log(f"{fam:30s} {r['raw_dev']:.4f} {r['dev']:.4f} {r['cv_mean']:.4f}±{r['cv_sd']:.4f}  {c['w_bm25']:6.2f} "
            f"{c['lam']:4.1f} {c['pen']:4.1f}  {r.get('no_bm25', float('nan')):7.4f} {r['no_col']:6.4f}")

# ---- test submissions (dev-best config of every family) ----
def write_zip(fam, split):
    r = results[fam]; c = r["config"]; S = SPLITS[split]
    X = score(split, r["sources"], c["w_bm25"], c["lam"], c["pen"])
    pred = {q: [S["pool"][j] for j in row] for q, row in zip(S["queries"], top7(X))}
    gold = {q: [] for q in S["all_queries"]} if split == "test" else S["rel"]
    errs = validate(pred, gold, set(S["pool"]))
    assert not errs, (fam, split, errs[:3])
    if split == "dev":
        tp = sum(len(set(pred[q]) & set(S["rel"][q])) for q in S["queries"])
        f1 = 2 * tp / (7 * len(pred) + sum(len(v) for v in S["rel"].values()))
        assert abs(f1 - r["dev"]) < 1e-9, (fam, f1, r["dev"])
        return
    d = os.path.join(OUT, "pred", fam); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "prediction.json"), "w", encoding="utf-8") as f: json.dump(pred, f)
    with zipfile.ZipFile(os.path.join(OUT, f"submission_{fam}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(d, "prediction.json"), "prediction.json")

for fam in results:
    write_zip(fam, "dev"); write_zip(fam, "test")
log(f"\nwrote {len(results)} test submissions to {OUT}/submission_<family>.zip (official format checks passed; "
    "dev predictions re-scored and equal to the table)")
json.dump(results, open(os.path.join(OUT, "pp_results.json"), "w"), indent=1)
open(os.path.join(OUT, "pp_report.txt"), "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
