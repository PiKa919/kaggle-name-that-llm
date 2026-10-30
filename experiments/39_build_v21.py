"""Patch v20 into v21: retune the two test-time-grouping knobs to their measured optimum.

Both changes come from experiments/36 and experiments/38 (findings.md §13), which swept base
accuracy against method, using the pipeline's own clustering view on real train logits:

  * `--adapt_thr` 0.15 -> 0.05.  Prompt recovery quality at the pipeline's own view:
    0.15 gives ARI 0.867 / mean cluster size 7.20; 0.05 gives ARI 0.988 / mean size 17.41.
    0.05 wins at every accuracy bin and every method tested, so this is safe on its own.

  * `--adapt_w` 1.0 -> 0.25.  At blend accuracy (base BA ~0.66-0.70) every w=1.0 centring
    variant measured -0.007 to -0.016, while w=0.25 measured +0.002 to +0.003. v20 ships the
    measured-worst setting of the whole grid.

No new members, no new data, no extra runtime. Predicted +0.005 to +0.015.
"""
import json
import pathlib
import sys

SRC = pathlib.Path("notebooks/v20/name-that-llm-v20.ipynb")
DST = pathlib.Path("notebooks/v21/name-that-llm-v21.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.load(open(SRC))


def join(cell):
    return "".join(cell["source"])


def patch(old, new, label):
    hits = [(i, join(c)) for i, c in enumerate(nb["cells"]) if old in join(c)]
    if len(hits) != 1:
        sys.exit(f"FAIL {label}: {len(hits)} occurrences")
    i, src = hits[0]
    nb["cells"][i]["source"] = src.replace(old, new).splitlines(keepends=True)
    print(f"ok {label}")


patch("'--adapt_w', '1.0',\n           '--adapt_thr', '0.15',",
      "'--adapt_w', '0.25',\n           '--adapt_thr', '0.05',",
      "final cmd: adapt_w 1.0->0.25, adapt_thr 0.15->0.05")

json.dump(nb, open(DST, "w"), indent=1)

meta = json.load(open("notebooks/v20/kernel-metadata.json"))
meta["id"] = "pikaaa/name-that-llm-v21-centre-retune"
meta["title"] = "name that llm v21 centre retune"
meta["code_file"] = "name-that-llm-v21.ipynb"
json.dump(meta, open(DST.parent / "kernel-metadata.json", "w"), indent=2)
print("written", DST)
