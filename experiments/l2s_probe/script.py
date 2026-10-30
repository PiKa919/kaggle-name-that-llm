"""Probe the public l2s3 Qwen3-8B pool-trained adapter on competition train.

The adapter/head were trained on an external pool (pool_v3), not on train.csv, so
train balanced accuracy is a clean generalization estimate (verify: adapter dir
has no train.csv, pool path is extdata/pool_v3.parquet).

For each checkpoint (stage-1 and full) and four pooling variants (all tokens vs
drop-first, with/without a trailing EOS), score 900 stratified train rows, pick the
best variant, then score all 2,699 rows and the 3-row demo test. Saves probabilities
for later ensemble integration and logs per-class recalls.

GPU only (Qwen3-8B fp16). No fitting, no labels used beyond scoring.
"""
import glob
import json
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

T0 = time.time()
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
L2I = {l: i for i, l in enumerate(LABELS)}
POOL_LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def one(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, pattern
    return hits[0]


data_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"]
                if os.path.exists(p + "/train.csv"))
train = pd.read_csv(data_dir + "/train.csv")
test = pd.read_csv(data_dir + "/test.csv")
test["text"] = test["text"].fillna("").astype(str)
y = train.model.map(L2I).values
groups = train.prompt_id.values
TR, TE = train.text.astype(str).tolist(), test.text.tolist()
log("train", train.shape, "test", test.shape)

# ---- base checkpoint: Qwen3 8B (hidden 4096, 36 layers)
base = None
for cfg_path in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
    try:
        c = json.load(open(cfg_path))
    except Exception:
        continue
    if c.get("model_type") == "qwen3" and c.get("hidden_size") == 4096 and c.get("num_hidden_layers") == 36:
        if any(os.path.exists(os.path.join(os.path.dirname(cfg_path), f))
               for f in ("model.safetensors", "model.safetensors.index.json",
                         "pytorch_model.bin", "pytorch_model.bin.index.json")):
            base = os.path.dirname(cfg_path)
            break
assert base, "Qwen3-8B-Base not attached"
log("base", base)

cfg = AutoConfig.from_pretrained(base)
cfg.use_cache = False
model = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained(base)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dev = model.get_input_embeddings().weight.device
log("model loaded, devices", {i: round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())})


def prep_adapter(cfg_file, weight_file, tag):
    d = f"/kaggle/working/adapter_{tag}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(cfg_file, os.path.join(d, "adapter_config.json"))
    shutil.copy(weight_file, os.path.join(d, "adapter_model.safetensors"))
    return d


def load_ckpt(stem):
    cfg_file = one(stem + "__adapter_config.json")
    weight_file = one(stem + "__adapter_model.safetensors")
    head_file = one(stem + "__head.pt")
    sd = torch.load(head_file, map_location="cpu")
    norms = nn.ModuleList([nn.LayerNorm(4096) for _ in POOL_LAYERS])
    norms.load_state_dict({k[len("norms."):]: v for k, v in sd.items() if k.startswith("norms.")})
    # Keep the norm/head on CPU float32: the base model is sharded across both T4s,
    # so hidden states arrive on different devices and CPU math is exact + trivial.
    return dict(dir=prep_adapter(cfg_file, weight_file, stem), norms=norms.eval(),
                W=sd["head.1.weight"].float(), b=sd["head.1.bias"].float(),
                Wc=sd.get("head_c.1.weight"), bc=sd.get("head_c.1.bias"), stem=stem)


@torch.no_grad()
def features(lm, texts, add_eos, norms, bs=8):
    """(n, 8*4096) layer-normed mean pools for both pooling variants: all tokens / drop first."""
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=448)["input_ids"]
    if add_eos:
        ids = [x + [tok.eos_token_id] for x in ids]
    ZA = np.zeros((len(ids), 4096 * len(POOL_LAYERS)), dtype=np.float32)
    ZN = np.zeros_like(ZA)
    order = np.argsort([len(x) for x in ids])
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        batch = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            out = lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        m = batch["attention_mask"].unsqueeze(-1).float()
        m0 = m.clone()
        m0[:, 0] = 0
        za, zn = [], []
        for i, L in enumerate(POOL_LAYERS):
            h = hs[L].float()
            a = (h * m).sum(1) / m.sum(1).clamp(min=1)
            n = (h * m0).sum(1) / m0.sum(1).clamp(min=1)
            za.append(norms[i](a.cpu()).numpy())
            zn.append(norms[i](n.cpu()).numpy())
        ZA[idx] = np.concatenate(za, 1)
        ZN[idx] = np.concatenate(zn, 1)
        del out, hs, batch
    return ZA, ZN


def head_probs(Z, W, b):
    Zp = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    logits = Zp @ W.cpu().numpy().T + b.cpu().numpy()
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(1, keepdims=True)


def ba(pred, yy):
    from sklearn.metrics import balanced_accuracy_score, recall_score
    r = recall_score(yy, pred, average=None)
    return float(balanced_accuracy_score(yy, pred)), r


rng = np.random.default_rng(42)
probe_idx = np.concatenate([rng.choice(np.where(y == c)[0], size=min(50, int((y == c).sum())), replace=False)
                            for c in range(18)])
probe_idx.sort()
log("probe rows", len(probe_idx))

results = []
for stem in ["ckpt__l2s3_q8_s1", "ckpt__l2s3_q8_full"]:
    ck = load_ckpt(stem)
    log(stem, "adapter", ck["dir"])
    lm = PeftModel.from_pretrained(model, ck["dir"]).eval()
    for add_eos in (False, True):
        ZA, ZN = features(lm, [TR[i] for i in probe_idx], add_eos, ck["norms"])
        for pool_name, Z in (("all", ZA), ("dropfirst", ZN)):
            P = head_probs(Z, ck["W"], ck["b"])
            pred = P.argmax(1)
            score, rec = ba(pred, y[probe_idx])
            name = f"{stem}|eos={int(add_eos)}|{pool_name}"
            log(f"PROBE {name}: BA {score:.4f} min {rec.min():.3f} median {np.median(rec):.3f}")
            results.append(dict(name=name, ba=score, min_recall=float(rec.min()), median_recall=float(np.median(rec))))
    model = lm.unload()
    del lm
    torch.cuda.empty_cache()

best = max(results, key=lambda r: r["ba"])
log("best probe", best["name"], best["ba"])
with open("/kaggle/working/l2s_probe_results.json", "w") as fh:
    json.dump(results, fh, indent=1)

# full train + demo test with the best variant
stem, eos_flag, pool_name = best["name"].split("|")
add_eos = eos_flag.endswith("1")
ck = load_ckpt(stem)
lm = PeftModel.from_pretrained(model, ck["dir"]).eval()

ZA, ZN = features(lm, TR, add_eos, ck["norms"])
P_tr = head_probs(ZA if pool_name == "all" else ZN, ck["W"], ck["b"])
pred = P_tr.argmax(1)
score, rec = ba(pred, y)
log(f"FULL TRAIN {best['name']}: BA {score:.4f} min {rec.min():.3f}")
for c, r in zip(LABELS, rec):
    log(f"  recall {c:24s} {r:.3f}")

ZT_A, ZT_N = features(lm, TE, add_eos, ck["norms"])
P_te = head_probs(ZT_A if pool_name == "all" else ZT_N, ck["W"], ck["b"])
np.savez("/kaggle/working/l2s3_probs.npz", train=P_tr, test=P_te, labels=np.array(LABELS),
         best=best["name"], train_ba=score)
log("saved l2s3_probs.npz; test argmax", np.array(LABELS)[P_te.argmax(1)].tolist())
log("done")
