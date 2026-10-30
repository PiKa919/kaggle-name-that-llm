"""
Build v23: Inject non-Qwen Architectural Diversity (OLMo-2 7B + Mistral 7B)

Replaces weak/slow redundant Qwen models (n14b at 0.332, q25_14 at 0.246) with:
1. olmo7: alexchilton/ntl-ts-olmo2-7b on allenai/olmo-2/transformers/1124-7b/1
2. mistral7: alexchilton/ntl-ts-mistral7b on mistral-ai/mistral/pytorch/7b-v0.1-hf/1

Retains:
- q14v5 (w=3.668), ll8v5 (w=3.602), q8v5 (w=1.863)
- svc4c memory-safe (capped 72k pool, 185k features)
- Gaussian head
- ATLAS 0.20 with margin gating
- Adaptive centering and Sinkhorn-Knopp balancing
"""
import json
import os
import pathlib
import shutil

SRC_DIR = pathlib.Path("notebooks/v22")
DST_DIR = pathlib.Path("notebooks/v23")
DST_DIR.mkdir(parents=True, exist_ok=True)

# 1. Update kernel metadata
with open(SRC_DIR / "kernel-metadata.json") as f:
    meta = json.load(f)

meta["id"] = "pikaaa/name-that-llm-v23-diverse-backbones"
meta["title"] = "name that llm v23 diverse backbones"
meta["code_file"] = "name-that-llm-v23.ipynb"

# Add OLMo-2 and Mistral datasets if not present
if "alexchilton/ntl-ts-olmo2-7b" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("alexchilton/ntl-ts-olmo2-7b")
if "alexchilton/ntl-ts-mistral7b" not in meta["dataset_sources"]:
    meta["dataset_sources"].append("alexchilton/ntl-ts-mistral7b")

# Add model sources
olmo_model = "allenai/olmo-2/Transformers/1124-7b/1"
mistral_model = "mistral-ai/mistral/PyTorch/7b-v0.1-hf/1"

if olmo_model not in meta["model_sources"]:
    meta["model_sources"].append(olmo_model)
if mistral_model not in meta["model_sources"]:
    meta["model_sources"].append(mistral_model)

with open(DST_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Updated kernel-metadata.json for v23")

# 2. Update notebook
with open(SRC_DIR / "name-that-llm-v22.ipynb") as f:
    nb = json.load(f)

# Patch Cell 3 to locate BOLMO, BMIST, AOLMO, AMIST
cell3_src = "".join(nb["cells"][3]["source"])

path_patch = """
# v23: Locate OLMo2-7B and Mistral-7B models and adapters
BOLMO = next((os.path.dirname(x) for x in cfgs if '/olmo-2/' in x.lower() and ('/1124-7b' in x.lower() or '/7b' in x.lower())), None)
BMIST = next((os.path.dirname(x) for x in cfgs if '/mistral/' in x.lower() and ('/7b' in x.lower() or 'v0.1' in x.lower())), None)

_olmo_heads = glob.glob('/kaggle/input/**/ntl-ts-olmo2-7b/heads.safetensors', recursive=True)
AOLMO = os.path.dirname(_olmo_heads[0]) if _olmo_heads else None

_mistral_heads = glob.glob('/kaggle/input/**/ntl-ts-mistral7b/heads.safetensors', recursive=True)
AMIST = os.path.dirname(_mistral_heads[0]) if _mistral_heads else None

print('v23 diverse models:', BOLMO, AOLMO, BMIST, AMIST)
"""

if "BOLMO" not in cell3_src:
    cell3_src = cell3_src + "\n" + path_patch
    nb["cells"][3]["source"] = cell3_src.splitlines(keepends=True)
    print("Patched Cell 3 with model discovery paths")

# Patch Cell 8 (MODELS definition)
cell8_src = "".join(nb["cells"][8]["source"])

old_models = """    MODELS = [
      {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.668, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.9140},
      {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 1.863, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.599},
      {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 3.602, "bs": 16, "max_len": 384, "twin": 0.35},
      {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 0.332, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
      {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 0.246, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},
      {"name": "deb", "type": "enc", "ckpt": DEB, "w": 0.344},
    ]"""

new_models = """    MODELS = [
      {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.668, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.9140},
      {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 1.863, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.599},
      {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 3.602, "bs": 16, "max_len": 384, "twin": 0.35},
    ]
    if BOLMO and AOLMO:
        MODELS.append({"name": "olmo7", "type": "l24", "path": BOLMO, "adapters": [AOLMO], "keep": 24, "layers": L, "w": 1.500, "bs": 16, "max_len": 384, "twin": 0.35})
    if BMIST and AMIST:
        MODELS.append({"name": "mistral7", "type": "l24", "path": BMIST, "adapters": [AMIST], "keep": 24, "layers": L, "w": 1.500, "bs": 16, "max_len": 384, "twin": 0.35})
    MODELS.append({"name": "deb", "type": "enc", "ckpt": DEB, "w": 0.344})"""

if old_models in cell8_src:
    cell8_src = cell8_src.replace(old_models, new_models)
    nb["cells"][8]["source"] = cell8_src.splitlines(keepends=True)
    print("Patched Cell 8 with v23 model ensemble")
else:
    print("WARNING: old_models signature not found in Cell 8, check formatting")

with open(DST_DIR / "name-that-llm-v23.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully generated {DST_DIR / 'name-that-llm-v23.ipynb'}")
