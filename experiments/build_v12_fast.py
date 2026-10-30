"""Build notebooks/v12_fast/script.py from the validated v13 source.

v12-fast is the v12 centre weighting (0.80 public twin, 0.20 claude twin) on the
v10 runtime envelope: curated-pool SVC, q25 dead-head removed, corrected label
remap.  It is deliberately a one-dict A/B against v13 (0.70 / 0.15).
"""
import ast
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "v13" / "script.py"
OUT = ROOT / "notebooks" / "v12_fast" / "script.py"
META_IN = ROOT / "notebooks" / "v13" / "kernel-metadata.json"
META_OUT = ROOT / "notebooks" / "v12_fast" / "kernel-metadata.json"

src = SRC.read_text()
OLD = """EXTRA_LORA24 = {'lora24_public': 0.10, 'lora24_public_twin': 0.70,
                'lora24_seed7': 0.05, 'lora24_seed7_twin': 0.0,
                'lora24_claude': 0.05, 'lora24_claude_twin': 0.15}"""
NEW = """EXTRA_LORA24 = {'lora24_public': 0.10, 'lora24_public_twin': 0.80,
                'lora24_seed7': 0.05, 'lora24_seed7_twin': 0.0,
                'lora24_claude': 0.05, 'lora24_claude_twin': 0.20}"""
assert src.count(OLD) == 1, "v13 EXTRA_LORA24 block not found exactly once"
src = src.replace(OLD, NEW, 1)
compile(src, str(OUT), "exec")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(src)

meta = json.loads(META_IN.read_text())
meta["id"] = "pikaaa/name-that-llm-v12"
meta["title"] = "name that llm v12 fast"
META_OUT.write_text(json.dumps(meta, indent=2) + "\n")
print("V12FAST_BUILD_OK", OUT, len(src))
print("META_OK", meta["id"], len(meta["dataset_sources"]), "datasets")
