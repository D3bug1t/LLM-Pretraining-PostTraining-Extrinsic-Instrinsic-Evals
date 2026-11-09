# Re-generate summaries locally from a Kaggle run's trained weights with other decoding settings (no retraining).
# Usage: python regen_phase6.py <run> <setting> [heldout|test|both]
#   settings: base (as on Kaggle), ng3 (no repeated 3-grams), rp115 (repetition penalty 1.15), min500 (>= 500 new tokens)
# Output: pp_work/regen/<run>__<setting>/heldout_generations.json (+ heldout_references.json), submission.zip
import os, re, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))     # bundle/, data/ and pp_work/ live next to this script
CAP = HERE
SETTINGS = {
    "base":   dict(),
    "ng3":    dict(no_repeat_ngram=3),
    "rp115":  dict(repetition_penalty=1.15),
    "min500": dict(min_new_tokens=500),
}
run, setting = sys.argv[1], sys.argv[2]
which = sys.argv[3] if len(sys.argv) > 3 else "heldout"
os.environ.setdefault("P6_LOCAL_DATA", os.path.join(HERE, "data", "Phase_6_Summ"))
os.environ.setdefault("P6_AMP", "fp16")                     # as on the Kaggle T4s

src = open(os.path.join(HERE, "summ_cells.py"), encoding="utf-8").read()
code = re.split(r"^# %%.*$", src, flags=re.M)[1:]
g = {"CAP": CAP, "os": os, "sys": sys, "__name__": "regen"}
exec(code[1], g)
exec(r'''
MODEL_DIRS.update({m: os.environ[m.upper() + "_DIR"] for m in ("m2", "m3", "m4", "m5") if m.upper() + "_DIR" in os.environ})
WORK = os.path.join(HERE, "pp_work", "regen_tmp")
''', {**g, "HERE": HERE})
g["HERE"] = HERE
for c in code[2:]:
    if "RUNNING" in c or "def write_summary" in c: break
    exec(c, g)
g.update(run=run, setting=setting, which=which, SETTINGS=SETTINGS, HERE=HERE)
exec(r'''
ck = torch.load(os.path.join(HERE, "bundle", run, "trainable_weights.pt"), map_location="cpu", weights_only=False)
cfg = {**ck["config"], **SETTINGS[setting]}
cfg["gen_batch"] = int(os.environ.get("GEN_BATCH", cfg["gen_batch"]))   # more documents per step: faster, same greedy output
refs = json.load(open(os.path.join(HERE, "bundle", run, "heldout_references.json"), encoding="utf-8"))
assert list(refs) == GEN_VAL_IDS, "held-out split differs from the Kaggle run"
model, info = build_model(cfg, cfg["student_layers"])
setup_method(model, cfg)
missing = model.load_state_dict({k: v.float() for k, v in ck["trainable"].items()}, strict=False)
assert not missing.unexpected_keys, missing.unexpected_keys
model.to(DEVICE).eval()
bpe, toks = get_tokens(info["tokenizer"])

_make_ban = make_ban
def make_ban(cfg):                          # + optional repetition penalty (CTRL: logit / p if > 0 else logit * p)
    base, p = _make_ban(cfg), cfg.get("repetition_penalty")
    def ban(lg, gens, done, step):
        if p and step:
            rows = [b for b, x in enumerate(gens) for _ in x]; cols = [t for x in gens for t in x]
            if rows:
                used = torch.zeros_like(lg, dtype=torch.bool); used[rows, cols] = True
                lg = torch.where(used, torch.where(lg > 0, lg / p, lg * p), lg)
        return base(lg, gens, done, step)
    return ban

out = os.path.join(HERE, "pp_work", "regen", f"{run}__{setting}"); os.makedirs(out, exist_ok=True)
gru = cfg["model"] == "m3"
if which in ("heldout", "both"):
    gv, s, hit = generate_texts(model, bpe, toks, GEN_VAL_IDS, cfg, gru)
    json.dump(gv, open(os.path.join(out, "heldout_generations.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(refs, open(os.path.join(out, "heldout_references.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{run} {setting} held-out: {np.mean([len(v.split()) for v in gv.values()]):.0f} words, {100 * hit:.0f}% hit max, {s:.0f}s")
if which in ("test", "both"):
    gt, s, hit = generate_texts(model, bpe, toks, TEST_IDS, cfg, gru)
    preds = [{"id": d, "summary": gt[d]} for d in TEST_IDS]
    assert len(preds) == 100 and all(p["summary"].strip() for p in preds)
    with zipfile.ZipFile(os.path.join(out, "submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(preds, ensure_ascii=False))
    print(f"{run} {setting} test: {np.mean([len(v.split()) for v in gt.values()]):.0f} words, {100 * hit:.0f}% hit max, {s:.0f}s")
''', g)
