# Phase 6 output-level ensemble (MBR / consensus selection): for each document, out of the candidate summaries of a pool,
# keep the one with the highest mean agreement with the other candidates. Agreement = mean of ROUGE-1 F1 and ROUGE-2 F1
# (rouge-score tokenizer with stemming). Uses only our own generations: no references, no external model.
# Usage: python ensemble_phase6.py      -> pp_work/ens/<pool>/{heldout_generations.json, heldout_references.json, submission.zip}
import json, os, zipfile
from collections import Counter
from rouge_score import tokenizers

HERE = os.path.dirname(os.path.abspath(__file__))
B, R = os.path.join(HERE, "bundle"), os.path.join(HERE, "pp_work", "regen")
TOK = tokenizers.DefaultTokenizer(use_stemmer=True)

def source(run, setting):
    """-> (held-out {id: summary}, test {id: summary}) for a run and decoding setting."""
    d = os.path.join(B, run) if setting == "base" else os.path.join(R, f"{run}__{setting}")
    held = json.load(open(os.path.join(d, "heldout_generations.json"), encoding="utf-8"))
    if setting == "base":
        test = json.load(open(os.path.join(d, "prediction.json"), encoding="utf-8"))
    else:
        with zipfile.ZipFile(os.path.join(d, "submission.zip")) as z: test = json.loads(z.read("prediction.json"))
    return held, {p["id"]: p["summary"] for p in test}

VARIANTS = ["base", "ng3", "rp115", "min500"]
POOLS = {
    "m4_lora_mbr":    [("m4_lora", s) for s in VARIANTS],
    "m2_houlsby_mbr": [("m2_houlsby", s) for s in VARIANTS],
    "all_mbr":        [(r, s) for r in ("m4_lora", "m2_houlsby") for s in VARIANTS]
                      + [("m5_lorma", "base"), ("m5_kd_student", "base"), ("m3_full", "base")],
}

def grams(t, n):
    return Counter(zip(*[t[i:] for i in range(n)]))

def f1(a, b):
    ov = sum((a & b).values()); na, nb = sum(a.values()), sum(b.values())
    return 0.0 if not ov else 2 * ov / (na + nb)

def select(cands):
    """cands: list of texts -> index of the candidate with the highest mean agreement with the others."""
    g = [(grams(t, 1), grams(t, 2)) for t in (TOK.tokenize(c) for c in cands)]
    score = [sum((f1(g[i][0], g[j][0]) + f1(g[i][1], g[j][1])) / 2 for j in range(len(g)) if j != i) for i in range(len(g))]
    return max(range(len(g)), key=score.__getitem__)

cache = {}
for name, pool in POOLS.items():
    srcs = [cache.setdefault(k, source(*k)) for k in pool]
    out = os.path.join(HERE, "pp_work", "ens", name); os.makedirs(out, exist_ok=True)
    for split in (0, 1):
        ids = list(srcs[0][split])
        assert all(list(s[split]) == ids or set(s[split]) == set(ids) for s in srcs)
        pick = {d: select([s[split][d] for s in srcs]) for d in ids}
        texts = {d: srcs[pick[d]][split][d] for d in ids}
        use = Counter(f"{pool[i][0]}/{pool[i][1]}" for i in pick.values())
        print(f"{name:15s} {'held-out' if split == 0 else 'test    '} picks: " + ", ".join(f"{k} {v}" for k, v in use.most_common()))
        if split == 0:
            json.dump(texts, open(os.path.join(out, "heldout_generations.json"), "w", encoding="utf-8"), ensure_ascii=False)
            json.dump(json.load(open(os.path.join(B, "m4_lora", "heldout_references.json"), encoding="utf-8")),
                      open(os.path.join(out, "heldout_references.json"), "w", encoding="utf-8"), ensure_ascii=False)
        else:
            preds = [{"id": d, "summary": texts[d]} for d in ids]
            assert len(preds) == 100 and all(p["summary"].strip() for p in preds)
            with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("prediction.json", json.dumps(preds, ensure_ascii=False))
