"""Figures and LaTeX tables for the final report, generated from the saved result files of every phase.
Run from anywhere: python Report/final/make_figures.py  -> Report/final/figures/*.png, Report/final/tables/*.tex"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG, TAB = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# one fixed colour per Phase 1 model (validated categorical palette, slots 1-4 in order); BM25 = neutral grey
COL = {"m2": "#2a78d6", "m3": "#eb6834", "m4": "#1baf7a", "m5": "#eda100", "bm25": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})

def load(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(CAP, pattern))):
        out += json.load(open(p, encoding="utf-8"))
    return out

def model_legend(ax, models, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL[m], label=m.upper() if m != "bm25" else "BM25") for m in models],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), fontsize=7, handlelength=1, columnspacing=1)

def hbars(ax, names, vals, models, fmt="{:.3f}"):
    y = range(len(names))
    ax.barh(list(y), vals, color=[COL[m] for m in models], height=.7, edgecolor="white", linewidth=1)
    ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i, v in zip(y, vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=6.5, color=INK2)

def tex_escape(s): return s.replace("_", r"\_")

def write_table(name, header, rows, align):
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r"\\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r"\\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

def params(n): return f"{n / 1e6:.2f}M" if n < 10e6 else f"{n / 1e6:.1f}M"

# ---------------------------------------------------------------- Phase 2 (values of the Phase 2 report tables)
p2 = [("m4_lorma_crf+vote+prop", "m4", .4813), ("m4_lorma_crf", "m4", .4502), ("m4_lorma_vit", "m4", .4131),
      ("m4_full", "m4", .3450), ("m4_lorma", "m4", .3202), ("m2_lora", "m2", .2899), ("m4_kd_student", "m4", .2841),
      ("m4_small_nokd", "m4", .2695), ("m3_full", "m3", .1906), ("m3_bitfit", "m3", .1149)]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
hbars(ax, [n for n, _, _ in p2], [v for _, _, v in p2], [m for _, m, _ in p2])
ax.set_xlabel("validation strict macro-F1 (10 documents)"); ax.set_xlim(0, .58)
model_legend(ax, ["m2", "m3", "m4"]); fig.savefig(os.path.join(FIG, "p2_runs.png")); fig.savefig(os.path.join(FIG, "p2_runs.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Phase 3 (dev macro-F1 of all runs)
p3 = [("m5_lora_crf", .4685), ("m5_full_crf", .4547), ("m5_rslora_crf", .4526), ("m5_rslora", .4499),
      ("m5_full", .4422), ("m5_lora", .4407), ("m4_lorma", .4381), ("m4_full", .4367), ("m4_lora", .4348),
      ("m4_rslora", .4329), ("m2_rslora", .4306), ("m5_lorma", .4308), ("m5_lorma_crf", .4272), ("m2_lorma", .4244),
      ("m2_lora", .4148), ("m5_kd_student", .3465), ("m3", .2052)]
p3.sort(key=lambda t: -t[1])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
hbars(ax, [n for n, _ in p3], [v for _, v in p3], [n[:2] for n, _ in p3])
ax.set_xlabel("dev macro-F1 (13 roles, 10 documents)"); ax.set_xlim(0, .55)
model_legend(ax, ["m2", "m3", "m4", "m5"]); fig.savefig(os.path.join(FIG, "p3_runs.png")); fig.savefig(os.path.join(FIG, "p3_runs.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Phase 4
p4 = load("Phase_4/phase4_results_bundle*/phase4_results_summary.json")
p4.sort(key=lambda r: -r["val_macro_f1"])
fig, ax = plt.subplots(figsize=(3.4, 3.2))
hbars(ax, [r["name"] for r in p4], [r["val_macro_f1"] for r in p4], [r["model"] for r in p4])
ax.set_xlabel("dev macro-F1 (994 cases, model's own reading, argmax)"); ax.set_xlim(0, .82)
model_legend(ax, ["m2", "m3", "m4", "m5"]); fig.savefig(os.path.join(FIG, "p4_runs.png")); fig.savefig(os.path.join(FIG, "p4_runs.pdf")); plt.close(fig)
write_table("p4_runs", ["Run", "Method", "Reads", "Trained", "Cases", "Ep.", "Dev F1", "Min."],
            [[r"\run{" + tex_escape(r["name"]) + "}", r["method"], f"{r['section']} {r['n_chunks']}$\\times${r['chunk_len']}",
              params(r["trainable_params"]), f"{r['train_cases'] / 1000:.1f}k", r["best_epoch"], f"{r['val_macro_f1']:.4f}",
              f"{r['train_minutes']:.0f}"] for r in p4], "lllrrrcr")
# which part of the case: dev F1 of the same trained model reading other parts (no retraining)
secs = ["head", "middle", "head_tail", "tail", "tail:8"]
rows = [r for r in p4 if r["model"] in ("m4", "m5") and r["train_cases"] == 10000 and r["section"] == "tail"
        and all(s in r["dev_sections"] for s in secs)]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
for r in rows:
    ax.plot(secs, [r["dev_sections"][s]["macro_f1"] for s in secs], marker="o", ms=4, lw=1.5,
            color=COL[r["model"]], alpha=.85)
ax.set_ylabel("dev macro-F1"); ax.set_xlabel("part of the case read at test time (trained on tail 512)")
model_legend(ax, ["m4", "m5"], loc="upper left"); fig.savefig(os.path.join(FIG, "p4_sections.png")); fig.savefig(os.path.join(FIG, "p4_sections.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Phase 5
pp5 = {}
for line in open(os.path.join(CAP, "Phase_5/pp_work/phase5_pp_log.txt"), encoding="utf-8"):
    t = line.split()
    if len(t) == 6 and t[1] in COL and t[0].endswith(("_global", "_perlabel")):
        pp5[t[0]] = (t[1], float(t[2]), float(t[3]))
p5 = load("Phase_5/bundle_*/phase5_results_summary.json")
p5.sort(key=lambda r: -r["val_macro_f1"])
write_table("p5_runs", ["Run", "Method", "Reads", "Trained", "Docs", "Ep.", "Dev F1", "Micro", "Thr."],
            [[r"\run{" + tex_escape(r["name"]) + "}", r["method"], f"{r['section']} {r['n_chunks']}$\\times${r['chunk_len']}",
              params(r["trainable_params"]), f"{r['train_docs'] / 1000:.0f}k" if r["train_docs"] else "all",
              r["best_epoch"], f"{r['val_macro_f1']:.4f}", f"{r['val_micro_f1']:.3f}", f"{r['val_threshold']:.2f}"]
             for r in p5], "lllrrrccc")
names = [k[:-len("_head4_global")] for k in pp5 if k.endswith("_head4_global")]
names.sort(key=lambda n: -pp5[n + "_head4_perlabel"][2])
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, n in enumerate(names):
    m, _, g = pp5[n + "_head4_global"]; _, _, pl = pp5[n + "_head4_perlabel"]
    ax.plot([g, pl], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter([g], [i], s=22, facecolor="white", edgecolor=COL[m], linewidth=1.5, zorder=2)
    ax.scatter([pl], [i], s=22, color=COL[m], zorder=3)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("dev macro-F1, 2-fold CV (open: one global threshold, filled: per-statute)")
model_legend(ax, ["m2", "m3", "m4", "m5"]); fig.savefig(os.path.join(FIG, "p5_thresholds.png")); fig.savefig(os.path.join(FIG, "p5_thresholds.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Phase 6
p6 = load("Phase_6/bundle/phase6_results_summary.json")
bs = {d["run"]: d for d in json.load(open(os.path.join(CAP, "Phase_6/pp_work/phase6_pp_summary.json")))
      if d["view"] == "trained input"}
p6.sort(key=lambda r: -bs[r["name"]]["BERTScore F1"])
write_table("p6_runs", ["Run", "Method", "Trained", "Ep.", "Val loss", "BERTScore", "R-1", "R-L", "Words", "Min."],
            [[r"\run{" + tex_escape(r["name"]) + "}", r["method"], params(r["trainable_params"]), f"{r['best_epoch']:.2f}".rstrip("0").rstrip("."),
              f"{r['val_loss']:.3f}", f"{bs[r['name']]['BERTScore F1']:.4f}", f"{bs[r['name']]['ROUGE-1']:.3f}",
              f"{bs[r['name']]['ROUGE-L']:.3f}", f"{bs[r['name']]['words']:.0f}", f"{r['train_minutes']:.0f}"] for r in p6],
            "llrrcccccr")
fig, ax = plt.subplots(figsize=(3.4, 1.7))
v6 = [bs[r["name"]]["BERTScore F1"] for r in p6]           # dots, not bars: the axis does not start at 0
ax.scatter(v6, range(len(p6)), s=36, color=[COL[r["model"]] for r in p6], zorder=3)
for i, v in enumerate(v6): ax.text(v, i - .32, f"{v:.4f}", ha="center", fontsize=6.5, color=INK2)
ax.set_yticks(range(len(p6))); ax.set_yticklabels([r["name"] for r in p6]); ax.invert_yaxis()
ax.grid(axis="y", visible=False); ax.set_ylim(len(p6) - .5, -.9)
ax.set_xlim(.79, .845); ax.set_xlabel("held-out BERTScore F1 (100 training documents never trained on)")
model_legend(ax, ["m2", "m3", "m4", "m5"], loc="lower left"); fig.savefig(os.path.join(FIG, "p6_bertscore.png")); fig.savefig(os.path.join(FIG, "p6_bertscore.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Phase 7
p7 = load("Phase_7/results/phase7_results_summary.json")
p7.sort(key=lambda r: -r["val_f1"])
write_table("p7_runs", ["Run", "Method", "Layers", "Trained", "Ep.", "0-shot", "Dev F1@7", "R@100", "Min."],
            [[r"\run{" + tex_escape(r["name"]) + "}", r["method"], r["layers"], params(r["trainable_params"]),
              r["best_epoch"], f"{r['zero_shot_f1']:.4f}", f"{r['val_f1']:.4f}", f"{r['val_recall_at_100']:.3f}",
              f"{r['train_minutes']:.0f}"] for r in p7], "lllrrcccr")
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for r in p7:
    h = r["history"]
    ax.plot([e["epoch"] for e in h], [e["val_f1"] for e in h], marker="o", ms=3, lw=1.5, color=COL[r["model"]],
            ls="--" if r["method"] in ("kd", "ia3") else "-", label=r["name"])
ax.axhline(.1857, color=COL["bm25"], lw=1, ls=":"); ax.text(0, .19, "BM25", fontsize=6, color=INK2)
ax.legend(fontsize=6, loc="lower right", ncol=2); ax.set_xlim(-.2, 6.2); ax.set_xlabel("epoch (0 = Phase 1 model, no training)"); ax.set_ylabel("dev Micro-F1@7")
fig.savefig(os.path.join(FIG, "p7_epochs.png")); fig.savefig(os.path.join(FIG, "p7_epochs.pdf")); plt.close(fig)

pp7 = json.load(open(os.path.join(CAP, "Phase_7/pp_work/pp_results.json")))
fams = ["bm25", "m3_full", "m4_full", "m5_full", "m5_lorma", "m5_lorma__spread4", "all_ens"]
steps = [("raw", lambda r: r["raw_dev"]), ("+ BM25, no column term", lambda r: r.get("no_col")),
         ("full post-processing (dev)", lambda r: r["dev"]), ("same, 2-fold CV", lambda r: r["cv_mean"])]
shades = ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]       # one hue, light -> dark = more steps applied
fig, ax = plt.subplots(figsize=(3.4, 2.3))
w = .2
for j, (lab, f) in enumerate(steps):
    xs = [i + (j - 1.5) * w for i in range(len(fams))]
    ax.bar(xs, [f(pp7[k]) or 0 for k in fams], width=w * .9, color=shades[j], label=lab, edgecolor="white", linewidth=.5)
ax.set_xticks(range(len(fams))); ax.set_xticklabels([k.replace("__", " ") for k in fams], fontsize=6.5, rotation=25, ha="right")
ax.set_ylabel("dev Micro-F1@7"); ax.grid(axis="x", visible=False); ax.legend(fontsize=6, loc="upper left", ncol=2)
ax.set_ylim(0, .52); fig.savefig(os.path.join(FIG, "p7_postprocess.png")); fig.savefig(os.path.join(FIG, "p7_postprocess.pdf")); plt.close(fig)
LB7 = {"bm25": .4306, "bm25__nopen": .4318, "all_ens": .4300, "all_ens__nopen": .4318, "m5_lorma__spread4": .4300,
       "m5_full": .4288, "m4_full": .4253, "m5_lorma__head1": .4259}           # Codabench, as submitted
lb = lambda k: f"{LB7[k]:.4f}" if k in LB7 else r"--"
rows = []
for k in ["bm25", "m3_full", "m4_full", "m4_ia3", "m5_full", "m5_lorma", "m5_lorma__head1", "m5_lorma__spread4", "m5_kd_student", "all_ens"]:
    a, b = pp7[k], pp7[k + "__nopen"]
    rows.append([r"\run{" + tex_escape(k) + "}", f"{a['raw_dev']:.4f}", f"{a['dev']:.4f}", f"{a['cv_mean']:.4f}",
                 f"{b['dev']:.4f}", f"{b['cv_mean']:.4f}", lb(k), lb(k + "__nopen")])
write_table("p7_pp", ["Base", "Raw", "Dev", "CV", "Dev$^-$", "CV$^-$", "LB", "LB$^-$"], rows, "lccccccc")
print("figures:", sorted(os.listdir(FIG))); print("tables:", sorted(os.listdir(TAB)))

# ---------------------------------------------------------------- per-class breakdowns (Phases 2, 3, 5)
import numpy as np, torch
v2 = json.load(open(os.path.join(CAP, "Phase_2/phase2_results_bundle_m5/m5_lorma_crf_pre/val_metrics.json")))["val_per_label"]
v2b = json.load(open(os.path.join(CAP, "Phase_2/phase2_results_bundle/m4_lorma/val_metrics.json")))["val_per_label"]
v3 = json.load(open(os.path.join(CAP, "Phase_3/phase3_results_bundle_m5_crf_kd/m5_lora_crf/val_metrics.json")))["val_per_label"]
dev5 = {str(r["id"]): r["labels"] for r in json.load(open(os.path.join(CAP, "Phase_5/data/Phase_5_LSI/Train/dev.json"), encoding="utf-8"))}
tr5 = json.load(open(os.path.join(CAP, "Phase_5/data/Phase_5_LSI/Train/train.json"), encoding="utf-8"))
freq5 = np.zeros(100)
for r in tr5:
    for l in (r["labels"] if isinstance(r["labels"], list) else json.loads(r["labels"])): freq5[l] += 1
L5 = [torch.load(os.path.join(CAP, f"Phase_5/bundle_A/{n}/logits.pt"))["dev"] for n in ("m4_full", "m4_lorma")]
ids5 = sorted(set(L5[0]) & set(L5[1]) & set(dev5))
P5 = np.mean([torch.sigmoid(torch.stack([L[i].float() for i in ids5])).numpy() for L in L5], 0)
Y5 = np.zeros_like(P5, dtype=bool)
for k, i in enumerate(ids5):
    for l in (dev5[i] if isinstance(dev5[i], list) else json.loads(dev5[i])): Y5[k, l] = True
G5 = np.round(np.arange(0.02, 0.92, 0.02), 2)
def f1s(Y, Z):
    tp = (Y & Z).sum(0); fp = (~Y & Z).sum(0); fn = (Y & ~Z).sum(0); return tp, fp, fn
perm = np.random.default_rng(0).permutation(len(ids5)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
TP = FP = FN = 0
for fit, held in ((A, B), (B, A)):                     # per-statute thresholds fitted on one half, counted on the other
    best = []
    for g in G5:
        tp, fp, fn = f1s(Y5[fit], P5[fit] >= g); best.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
    t = G5[np.argmax(np.stack(best), 0)]
    Z = P5[held] >= t; Z[np.arange(len(held)), P5[held].argmax(1)] = True
    tp, fp, fn = f1s(Y5[held], Z); TP, FP, FN = TP + tp, FP + fp, FN + fn
f5 = 2 * TP / np.maximum(2 * TP + FP + FN, 1)
json.dump(dict(p5_per_statute_f1=f5.tolist(), p5_train_freq=freq5.tolist(), p5_macro=float(f5.mean())),
          open(os.path.join(TAB, "p5_per_statute.json"), "w"))

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1.05, 1.15, 1]))
a = ax[0]; labs = list(v2); y = np.arange(len(labs))
a.barh(y - .2, [v2b[l] for l in labs], .4, color="#86b6ef", label="M4 LoRMA, softmax")
a.barh(y + .2, [v2[l] for l in labs], .4, color=COL["m5"], label="M5 LoRMA, CRF, preamble")
a.set_yticks(y); a.set_yticklabels(labs, fontsize=6.5); a.invert_yaxis(); a.grid(axis="y", visible=False)
a.set_xlim(0, 1); a.set_xlabel("validation F1"); a.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(.4, -.25), ncol=1)
a.set_title("(a) Phase 2 per entity type", fontsize=7.5, loc="left")
a = ax[1]; rl = sorted(v3, key=lambda r: -v3[r]); y = np.arange(len(rl))
a.barh(y, [v3[r] for r in rl], .7, color=COL["m5"]); a.set_yticks(y); a.set_yticklabels(rl, fontsize=6); a.invert_yaxis()
a.grid(axis="y", visible=False); a.set_xlim(0, 1); a.set_xlabel("dev F1 (m5_lora_crf)")
a.set_title("(b) Phase 3 per role", fontsize=7.5, loc="left")
a = ax[2]; a.scatter(freq5, f5, s=9, color=COL["m4"], alpha=.8, linewidths=0); a.set_xscale("log")
a.set_xlabel("training documents with the statute (log)"); a.set_ylabel("dev F1 (2-fold CV)")
a.set_title(f"(c) Phase 5 per statute (M4 ens.)", fontsize=7.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "per_class.png")); fig.savefig(os.path.join(FIG, "per_class.pdf"))
print("P5 per-statute macro (2-fold, one split):", round(float(f5.mean()), 4), "| statutes with F1 < 0.1:", int((f5 < .1).sum()))
