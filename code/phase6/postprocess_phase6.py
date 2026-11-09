# Phase 6 post-processing: score every run's held-out summaries with the official metrics (evaluate_phase6.py:
# BERTScore F1 with lang="en", ROUGE-1 F1, ROUGE-L F1) and collect the submission zips in pp_work/.
# Usage: python postprocess_phase6.py [bundle dir]   (default: Phase_6/bundle; unzip phase6_results_bundle.zip there)
import glob, json, os, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
B = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bundle")
OUT = os.path.join(HERE, "pp_work")
os.makedirs(OUT, exist_ok=True)
EVAL = os.path.join(HERE, "evaluate_phase6.py")   # official evaluator, copied next to this script
truth = os.path.join(B, "heldout_references.json")
LOG = open(os.path.join(OUT, "phase6_pp_log.txt"), "w", encoding="utf-8")
def log(s=""):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def evaluate(pred):
    t = os.path.join(os.path.dirname(pred), "heldout_references.json")    # the run's own references
    r = subprocess.run([sys.executable, EVAL, "--truth", t if os.path.exists(t) else truth, "--prediction", pred],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode: return dict(error=r.stderr.strip()[-300:])
    m = dict(l.split(": ") for l in r.stdout.strip().splitlines() if ": " in l)
    return {k: float(v) for k, v in m.items() if k != "Documents"}

rows = []
for f in sorted(glob.glob(os.path.join(B, "*", "heldout_generations*.json"))):
    run = os.path.basename(os.path.dirname(f))
    view = os.path.basename(f)[len("heldout_generations"):-len(".json")].lstrip("_") or "trained input"
    m = evaluate(f)
    words = sum(len(s.split()) for s in json.load(open(f, encoding="utf-8")).values()) / max(1, len(json.load(open(f, encoding="utf-8"))))
    rows.append(dict(run=run, view=view, words=words, **m))
    log(f"{run:16s} {view:14s} " + (f"BERTScore F1 {m['BERTScore F1']:.4f}  ROUGE-1 {m['ROUGE-1']:.4f}  ROUGE-L {m['ROUGE-L']:.4f}"
                                     if "error" not in m else f"FAILED {m['error']}") + f"  {words:.0f} words")
    sub = os.path.join(B, run, "submission.zip")
    if view == "trained input" and os.path.exists(sub):
        with zipfile.ZipFile(sub) as z:
            p = json.loads(z.read("prediction.json"))
        assert z.namelist() == ["prediction.json"] and all(x["summary"].strip() for x in p), sub
        shutil.copy(sub, os.path.join(OUT, f"submission_{run}.zip"))
refs = json.load(open(truth, encoding="utf-8"))
log(f"\nheld-out references: {len(refs)} documents, {sum(len(s.split()) for s in refs.values()) / len(refs):.0f} words")
json.dump(rows, open(os.path.join(OUT, "phase6_pp_summary.json"), "w"), indent=1)
log(f"zips in {OUT}: {sorted(os.path.basename(z) for z in glob.glob(os.path.join(OUT, 'submission_*.zip')))}")
