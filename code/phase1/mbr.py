"""Minimum Bayes Risk (MBR) decoding on top of infer.py.

For every prompt a pool of candidates is generated (deterministic greedy/beam variants + seeded samples),
then the candidate with the highest average utility against the rest of the pool is chosen
(the "consensus" continuation). Utilities:
  bert   : BERTScore F1 exactly as bert_score computes it for lang="en" (roberta-large, layer 17,
           no idf, no rescaling); each candidate is embedded once, pairs are cheap matrix products
  rouge  : ROUGE-L F1 (evaluate_phase1.rouge_l_f1)
  mix    : mean of the two

Usage:
  python mbr.py --split pseudo --n 300     # build pool, score every utility vs. the references
  python mbr.py --split test --utility bert # full test set -> submission_mbr_bert.zip
The pool is cached in outputs/mbr_pool_{split}.json, so changing --utility does not regenerate.
"""
import os, json, time, argparse, zipfile, random
import torch
import torch.nn.functional as F
import infer
from infer import STRATEGIES, DEFAULTS, OUT_DIR, HERE, load_model, load_split, BPETokenizer, MODEL_DIR

# (preset, seed) pairs; deterministic presets ignore the seed
POOL = [("greedy_ng4_rp11", 0), ("greedy_ng4_rp", 0), ("greedy_ng3_rp11", 0), ("greedy_ng3", 0),
        ("beam4_ng3", 0), ("beam2_ng3_rp11", 0)] + \
       [("topp_t07_rp11", s) for s in (1, 2, 3, 4)] + [("minp_t05_rp11", s) for s in (1, 2, 3, 4)]


def build_pool(data, cli):
    path = os.path.join(OUT_DIR, f"mbr_pool_{cli.split}_{len(data)}.json")
    part = path + ".partial"
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    done = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else {"n_done": 0, "pool": None}
    device = torch.device("cuda")
    model, cfg = load_model(device)
    tok = BPETokenizer(os.path.join(MODEL_DIR, "bpe_tokenizer.json"))
    pool = done["pool"] or {k: [] for k, _, _ in data}
    for name, seed in POOL[done["n_done"]:]:
        stype, settings = STRATEGIES[name]
        args = argparse.Namespace(**{**DEFAULTS, **settings, "max_new_tokens": 384, "target_words": 128})
        args.strategy_type = stype
        torch.manual_seed(seed); random.seed(seed)
        bs = 32 // args.num_beams if stype == "beam" else 32
        fn = infer.generate_beam if stype == "beam" else infer.generate_standard
        t0 = time.time()
        for s in range(0, len(data), bs):
            batch = data[s:s + bs]
            for (k, _, _), g in zip(batch, fn(model, tok, [p for _, p, _ in batch], cfg, args, device)):
                pool[k].append(" ".join(tok.decode(g).split()[:128]) or "the")
            if (s // bs) % 5 == 0:
                print(f"    {name}: {s + len(batch)}/{len(data)} {time.time() - t0:.0f}s", flush=True)
        print(f"  pool: {name} seed {seed} done in {time.time() - t0:.0f}s", flush=True)
        done = {"n_done": done["n_done"] + 1, "pool": pool}
        json.dump(done, open(part, "w", encoding="utf-8"), ensure_ascii=False)   # resumable
    del model
    torch.cuda.empty_cache()
    json.dump(pool, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    os.remove(part)
    return pool


class BertEmbedder:
    """Token embeddings as bert_score uses them for lang='en'."""

    def __init__(self, device):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModel.from_pretrained("roberta-large").to(device).eval().half()
        self.model.encoder.layer = self.model.encoder.layer[:17]       # bert_score: num_layers=17
        self.device = device

    @torch.inference_mode()
    def embed(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([t.strip() for t in texts[i:i + bs]], padding=True, truncation=True,
                           max_length=510, return_tensors="pt", return_special_tokens_mask=True).to(self.device)
            special = enc.pop("special_tokens_mask").bool()
            h = self.model(**enc).last_hidden_state.float()
            h = F.normalize(h, dim=-1)
            keep = enc["attention_mask"].bool() & ~special             # CLS / SEP / pad weigh 0
            for r in range(h.size(0)):
                out.append(h[r][keep[r]])
        return out


def bert_matrix(embs):
    """(C, C) BERTScore F1 of candidate i (prediction) vs j (reference); symmetric."""
    C = len(embs)
    Lm = max(e.size(0) for e in embs)
    E = torch.zeros(C, Lm, embs[0].size(1), device=embs[0].device)
    M = torch.zeros(C, Lm, dtype=torch.bool, device=embs[0].device)
    for i, e in enumerate(embs):
        E[i, :e.size(0)] = e
        M[i, :e.size(0)] = True
    S = torch.einsum("ild,jmd->ijlm", E, E)                             # (C, C, Li, Lj)
    S = S.masked_fill(~M[None, :, None, :], -2).masked_fill(~M[:, None, :, None], -2)
    P = (S.max(-1).values * M[:, None, :]).sum(-1) / M.sum(-1)[:, None]   # precision of i vs j
    R = P.T
    return 2 * P * R / (P + R)


def select(pool_k, utility, embs_k=None):
    from evaluate_phase1 import rouge_l_f1
    C = len(pool_k)
    u = torch.zeros(C, C)
    if utility in ("rouge", "mix"):
        r = torch.tensor([[rouge_l_f1(a, b) if i != j else 0.0 for j, b in enumerate(pool_k)]
                          for i, a in enumerate(pool_k)])
        u += r / (r.sum() / (C * (C - 1)))                                # scale-free when mixed
    if utility in ("bert", "mix"):
        b = bert_matrix(embs_k).cpu()
        b.fill_diagonal_(0)
        u += b / (b.sum() / (C * (C - 1)))
    return int(u.sum(1).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["pseudo", "test"], default="pseudo")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--utility", nargs="+", default=["bert", "rouge", "mix"], choices=["bert", "rouge", "mix"])
    cli = ap.parse_args()
    data = load_split(cli.split, cli.n)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    pool = build_pool(data, cli)
    print(f"pool ready: {len(pool)} prompts x {len(POOL)} candidates ({time.time() - t0:.0f}s)", flush=True)

    embs = None
    if any(u in ("bert", "mix") for u in cli.utility):
        emb = BertEmbedder(torch.device("cuda"))
        keys = [k for k, _, _ in data]
        flat = emb.embed([c for k in keys for c in pool[k]])
        C = len(POOL)
        embs = {k: flat[i * C:(i + 1) * C] for i, k in enumerate(keys)}
        del emb
        torch.cuda.empty_cache()

    for utility in cli.utility:
        t1 = time.time()
        choice = {k: select(pool[k], utility, embs[k] if embs else None) for k, _, _ in data}
        preds = {k: {"text": pool[k][choice[k]]} for k, _, _ in data}
        picked = [POOL[c][0] for c in choice.values()]
        mix = {n: picked.count(n) for n in dict.fromkeys(picked)}
        name = f"mbr_{utility}"
        row = {"strategy": name, "split": cli.split, "n": len(data), "select_seconds": round(time.time() - t1, 1),
               "picked_from": mix}
        if cli.split == "pseudo":
            row.update(infer.score([preds[k]["text"] for k, _, _ in data], [r for _, _, r in data], True))
            # oracle: best candidate per prompt by BERTScore vs the reference (upper bound for MBR)
        else:
            d = os.path.join(OUT_DIR, name)
            os.makedirs(d, exist_ok=True)
            pj = os.path.join(d, "prediction.json")
            json.dump(preds, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            with zipfile.ZipFile(os.path.join(HERE, f"submission_{name}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
                z.write(pj, arcname="prediction.json")
        print(json.dumps(row), flush=True)
        with open(os.path.join(OUT_DIR, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
