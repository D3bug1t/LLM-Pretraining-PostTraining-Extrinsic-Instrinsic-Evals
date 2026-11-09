"""Data statistics for the report (Section 3): one overview table + a 4-panel figure, computed from the official files.
python Report/final/data_stats.py  -> tables/data_overview.tex, tables/data_facts.json, figures/data_panels.pdf"""
import json, os, re, statistics as st
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
J = lambda *p: json.load(open(os.path.join(CAP, *p), encoding="utf-8"))
words = lambda s: len(s.split())
med = lambda xs: int(np.median(xs)); p90 = lambda xs: int(np.percentile(xs, 90))
F = {}

# Phase 1: test prompts (corpus sizes from the Phase 1 data listing)
p1 = J("Inference", "data", "test.json")
F["p1"] = dict(prompts=len(p1), words_median=med([words(v["text"]) for v in p1.values()]))

# Phase 2: from the Phase 2 data analysis (fold1 + fold2 = training, fold3 = test)
eda = J("Phase_2", "data_analysis_phase_2", "phase2_eda_summary.json")
lab2 = Counter(eda["label_counts"]["fold1"]) + Counter(eda["label_counts"]["fold2"])
F["p2"] = dict(train_docs=70, test_docs=35, entities=sum(lab2.values()), labels=dict(lab2),
               words_median=st.median([eda["lengths"][f]["words_median"] for f in ("fold1", "fold2")]))

# Phase 3
roles = {d: Counter() for d in ("CL", "IT")}; s3 = {}; sl = []
for d in ("CL", "IT"):
    for part in ("train", "dev"):
        docs = J("Phase_3", "data", "Phase_3_RR", "Train", f"{d}_{part}.json")
        s3[f"{d}_{part}"] = (len(docs), sum(len(x["text"]) for x in docs))
        if part == "train":
            for x in docs: roles[d].update(x["labels"]); sl += [words(s) for s in x["text"]]
    t = J("Phase_3", "data", "Phase_3_RR", "Test", f"{d}_test.json"); s3[f"{d}_test"] = (len(t), sum(len(v) for v in t.values()))
F["p3"] = dict(splits=s3, sent_words_median=med(sl), roles={d: dict(c) for d, c in roles.items()})

# Phase 4
tr, dv, te = (J("Phase_4", "data", "Phase_4_CJPE", *p) for p in (("Train", "train.json"), ("Train", "dev.json"), ("Test", "test.json")))
uniq = {x["text"]: x["label"] for x in tr}
test_texts = {x["text"] for x in te}
dev_only = {x["text"]: x["label"] for x in dv if x["text"] not in test_texts}
w4 = [words(t) for t in uniq]
F["p4"] = dict(train_rows=len(tr), train_unique=len(uniq), train_pos=float(np.mean(list(uniq.values()))),
               dev_rows=len(dv), dev_in_test=sum(x["text"] in test_texts for x in dv), dev_only=len(dev_only),
               dev_pos=float(np.mean(list(dev_only.values()))), test=len(te), words_median=med(w4), words_p90=p90(w4))

# Phase 5
tr5, dv5, te5 = (J("Phase_5", "data", "Phase_5_LSI", *p) for p in (("Train", "train.json"), ("Train", "dev.json"), ("Test", "test.json")))
lab = lambda x: x["labels"] if isinstance(x["labels"], list) else json.loads(x["labels"])
per_doc = [len(lab(x)) for x in tr5]; freq5 = Counter(l for x in tr5 for l in lab(x))
w5 = [words(x["text"]) for x in tr5]
F["p5"] = dict(train=len(tr5), dev=len(dv5), test=len(te5), statutes=len(freq5), labels_per_doc=float(np.mean(per_doc)),
               rarest=min(freq5.values()), most=max(freq5.values()), words_median=med(w5), words_p90=p90(w5))

# Phase 6
tr6, te6 = J("Phase_6", "data", "Phase_6_Summ", "Train", "train.json"), J("Phase_6", "data", "Phase_6_Summ", "Test", "test.json")
dw, sw, ext = [], [], []
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
for x in tr6:
    d, s = tok(x["document"]), tok(x["summary"]); dw.append(len(d)); sw.append(len(s))
    db = set(zip(d, d[1:])); sb = list(zip(s, s[1:]))
    ext.append(np.mean([b in db for b in sb]) if sb else 0)
F["p6"] = dict(train=len(tr6), test=len(te6), doc_words_median=med(dw), doc_words_p90=p90(dw), sum_words_median=med(sw),
               compression_median=float(np.median(np.array(dw) / np.maximum(1, sw))), bigram_overlap_median=float(np.median(ext)),
               test_doc_words_median=med([len(tok(x["document"])) for x in te6]))

# Phase 7
p7 = {}
for s, f in (("train", "Train"), ("dev", "Dev"), ("test", "Test")):
    q, c = J("Phase_7", "data", "Phase_7_PCR", f, f"{s}.json"), J("Phase_7", "data", "Phase_7_PCR", f, f"{s}_candidates.json")
    rel = [len([y for y in v.get("relevant_candidates", []) if y.strip()]) for v in q.values()]
    p7[s] = dict(queries=len(q), pool=len(c), rel_mean=float(np.mean(rel)) if s != "test" else None,
                 words_median=med([words(" ".join(v["text"])) for v in c.values()]))
F["p7"] = p7
json.dump(F, open(os.path.join(TAB, "data_facts.json"), "w"), indent=1)

# ---- overview table ----
fmt = lambda n: f"{n:,}".replace(",", "{,}")
rows = [
    ("1 LM", "1.3\\,GB SC text (of 19.2\\,GB)", "---", f"{fmt(F['p1']['prompts'])} prompts", "512 / 128 words", "BERTScore F1"),
    ("2 NER", f"{F['p2']['train_docs']} docs, {fmt(F['p2']['entities'])} spans", "10 docs (held out)", f"{F['p2']['test_docs']} docs",
     f"{fmt(int(F['p2']['words_median']))} words", "strict macro-F1"),
    ("3 RR", f"80 docs, {fmt(s3['CL_train'][1] + s3['IT_train'][1])} sent.", f"10 docs, {fmt(s3['CL_dev'][1] + s3['IT_dev'][1])} sent.",
     f"10 docs, {fmt(s3['CL_test'][1] + s3['IT_test'][1])} sent.", f"{F['p3']['sent_words_median']} words/sent.", "macro-F1"),
    ("4 CJPE", f"{fmt(F['p4']['train_unique'])} cases", f"{fmt(F['p4']['dev_only'])} cases$^\\dagger$", f"{fmt(F['p4']['test'])} cases",
     f"{fmt(F['p4']['words_median'])} words", "macro-F1"),
    ("5 LSI", f"{fmt(F['p5']['train'])} docs", f"{fmt(F['p5']['dev'])} docs", f"{fmt(F['p5']['test'])} docs",
     f"{fmt(F['p5']['words_median'])} words", "macro-F1 (100)"),
    ("6 Summ.", f"{fmt(F['p6']['train'])} pairs", "200 held out", f"{F['p6']['test']} docs",
     f"{fmt(F['p6']['doc_words_median'])} $\\to$ {F['p6']['sum_words_median']}", "BERTScore F1"),
    ("7 PCR", f"{p7['train']['queries']} q / {fmt(p7['train']['pool'])} c", f"{p7['dev']['queries']} q / {fmt(p7['dev']['pool'])} c",
     f"{p7['test']['queries']} q / {fmt(p7['test']['pool'])} c", f"{fmt(p7['test']['words_median'])} words", "Micro-F1@7"),
]
lines = [r"\begin{tabular}{llllll}", r"\toprule", r"Phase & Train & Dev & Test & Median length & Metric\\", r"\midrule"]
lines += [" & ".join(r) + r"\\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "data_overview.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---- 4-panel figure ----
INK2, GRID, BLUE, ORANGE = "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.6))
a = ax[0, 0]; L = sorted(lab2, key=lambda k: -lab2[k])
a.bar(range(len(L)), [lab2[k] for k in L], color=BLUE, width=.75); a.set_xticks(range(len(L)))
a.set_xticklabels([k.replace("COUNSEL", "C.") for k in L], rotation=45, ha="right"); a.grid(axis="x", visible=False)
a.set_title("(a) Phase 2: gold spans per entity type (70 training docs)", fontsize=7.5, loc="left")
a = ax[0, 1]; R = sorted(set(roles["CL"]) | set(roles["IT"]), key=lambda r: -(roles["CL"][r] + roles["IT"][r]))
nC, nI = sum(roles["CL"].values()), sum(roles["IT"].values()); y = np.arange(len(R))
a.barh(y - .2, [100 * roles["CL"][r] / nC for r in R], .4, color=BLUE, label=f"CL ({fmt(nC).replace('{,}', ',')} sent.)")
a.barh(y + .2, [100 * roles["IT"][r] / nI for r in R], .4, color=ORANGE, label=f"IT ({fmt(nI).replace('{,}', ',')} sent.)")
a.set_yticks(y); a.set_yticklabels(R, fontsize=6.5); a.invert_yaxis(); a.grid(axis="y", visible=False)
a.set_xlabel("% of training sentences"); a.legend(fontsize=6.5, loc="lower right")
a.set_title("(b) Phase 3: rhetorical roles by domain", fontsize=7.5, loc="left")
a = ax[1, 0]; fr = sorted(freq5.values(), reverse=True)
a.plot(range(1, len(fr) + 1), fr, color=BLUE, lw=1.8); a.set_yscale("log")
a.set_xlabel("statute rank"); a.set_ylabel("training documents (log)")
a.set_title(f"(c) Phase 5: statute frequency ({F['p5']['labels_per_doc']:.1f} labels/doc)", fontsize=7.5, loc="left")
a = ax[1, 1]
a.scatter(dw, sw, s=3, color=BLUE, alpha=.25, linewidths=0); a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("document words (log)"); a.set_ylabel("summary words (log)")
a.set_title(f"(d) Phase 6: document vs. summary words (median ratio {F['p6']['compression_median']:.1f})", fontsize=7.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "data_panels.pdf")); fig.savefig(os.path.join(FIG, "data_panels.png"), dpi=200)
print(json.dumps({k: v for k, v in F.items() if k != "p3"}, indent=1)[:2500]); print("p3 splits", s3)
