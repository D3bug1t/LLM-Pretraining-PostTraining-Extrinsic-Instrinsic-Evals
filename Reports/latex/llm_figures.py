"""Two figures for the LLM-based analysis (llm_analysis/llm_based_results_and_data_analysis.tex).
Values are the local / Codabench pairs and distillation results reported in the main report's tables.
Run: python Report/final/llm_figures.py  -> Report/final/figures/llm_calibration.pdf, llm_kd.pdf"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})

# (local estimate, Codabench) per submission; local = CV where a decision was fitted on validation data
PAIRS = {
    "P1 LM (pseudo-val.)": [(.8327, .8324), (.8355, .8343), (.8353, .8345), (.8369, .8352), (.8372, .8363),
                            (.8393, .8379), (.8419, .8407)],
    "P2 NER (10 val. docs)": [(.3202, .3442), (.2899, .3416), (.4813, .4715)],
    "P3 RR (dev)": [(.439, .4952), (.350, .3338), (.441, .5245), (.463, .5620), (.472, .5430), (.496, .5590),
                    (.453, .5040), (.471, .5713), (.474, .5557), (.469, .5657)],
    "P4 CJPE (CV)": [(.6924, .6941), (.7188, .6889), (.7537, .7374), (.7521, .7609), (.7484, .7523)],
    "P5 LSI (CV)": [(.3673, .3664), (.3624, .3588), (.3502, .3476), (.3441, .3461)],
    "P6 Summ. (held-out)": [(.8423, .8381), (.8414, .8380), (.8378, .8312), (.8377, .8329), (.8361, .8322)],
    "P7 PCR (CV)": [(.4264, .4306), (.4249, .4318), (.4280, .4253), (.4311, .4288), (.4331, .4259), (.4317, .4300),
                    (.4284, .4300), (.4245, .4318)],
}
fig, ax = plt.subplots(figsize=(5.2, 2.5))
names = list(PAIRS)
for i, k in enumerate(names):
    d = [lb - loc for loc, lb in PAIRS[k]]
    ax.scatter(d, [i] * len(d), s=26, color=BLUE, edgecolor="white", linewidth=.6, zorder=3,
               label="Codabench $-$ local" if i == 0 else None)
    if k.startswith("P3"):
        r = [lb - loc * 13 / 11 for loc, lb in PAIRS[k]]
        ax.scatter(r, [i - .28] * len(r), s=26, facecolor="white", edgecolor=ORANGE, linewidth=1.2, zorder=3,
                   label="P3 with dev rescaled by 13/11")
ax.axvline(0, color=INK2, linewidth=.8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("Codabench score minus local estimate (one dot per submission)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7, handletextpad=.3)
fig.savefig(os.path.join(FIG, "llm_calibration.pdf")); fig.savefig(os.path.join(FIG, "llm_calibration.png")); plt.close(fig)

# distillation: student score as a share of its teacher's (same split, same metric)
KD = [("P4 CJPE  (m5_full)", .6503 / .6528), ("P5 LSI  (m5_full)", .3460 / .3531), ("P7 PCR  (m5_lorma)", .2913 / .3091),
      ("P2 NER  (m4_full)", .2841 / .3450), ("P3 RR  (m5_full)", .3465 / .4422)]
fig, ax = plt.subplots(figsize=(4.2, 1.7))
y = range(len(KD))
ax.barh(list(y), [100 * v for _, v in KD], color=BLUE, height=.6, edgecolor="white", linewidth=1)
for i, (_, v) in zip(y, KD):
    ax.text(100 * v, i, f" {100 * v:.1f}%", va="center", fontsize=7, color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels([n for n, _ in KD]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(0, 112); ax.set_xlabel("4-layer student's validation score as % of its teacher's (teacher in brackets)")
fig.savefig(os.path.join(FIG, "llm_kd.pdf")); fig.savefig(os.path.join(FIG, "llm_kd.png")); plt.close(fig)
print("written llm_calibration, llm_kd")
