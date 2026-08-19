"""Check every backbone feature file: image count, dimensionality, L2 norm.

    uv run python tools/check_backbones.py

A backbone is usable only if it covers all 6526 stimuli and its rows are
L2-normalised (Ryu & Yanaka). Anything else is reported as FAIL with the
reason, so a half-finished extraction cannot quietly become a table row.

Fine-tuned files also carry best_epoch/history, which is printed so early
stopping can be seen to have worked rather than assumed.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_STIMULI = 6526

BACKBONES = {
    "CLIP frozen":       ["features/clip_features.npz"],
    "Qwen3-VL 8B":       ["features/vlm_LT15.npz"],
    "Qwen3-VL 4B":       ["features/vlm4b_LT15.npz"],
    "CLIP-ft (emotion)": ["features/clip_ftpf_emotion_v4_results/clip_ftpf_emotion_v4_fold*.npz"],
    "CLIP-ft (score)":   ["features/clip_ftpf_overall_v4_results/clip_ftpf_overall_v4_fold*.npz"],
}


def check(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    x = d["features"]
    n = np.linalg.norm(x, axis=1)
    r = {"n_img": x.shape[0], "dim": x.shape[1],
         "norm_lo": float(n.min()), "norm_hi": float(n.max())}
    r["normalised"] = abs(n.mean() - 1) < 1e-3 and n.std() < 1e-3
    r["complete"] = x.shape[0] == N_STIMULI
    r["ids_unique"] = len(set(map(str, d["stimulus_ids"]))) == x.shape[0]
    if "best_epoch" in d.files:
        r["epoch"] = f"best={int(d['best_epoch'])}/ran={len(d['history'])}"
    return r


def main() -> int:
    bad = 0
    for name, patterns in BACKBONES.items():
        files = sorted(p for pat in patterns
                       for p in glob.glob(str(ROOT / pat)) if "backup" not in p)
        if not files:
            print(f"{name:20s} MISSING (no files)")
            bad += 1
            continue
        for f in files:
            r = check(Path(f))
            ok = r["complete"] and r["normalised"] and r["ids_unique"]
            why = "" if ok else " <- " + ", ".join(filter(None, [
                None if r["complete"] else f"only {r['n_img']}/{N_STIMULI} images",
                None if r["normalised"] else f"not L2-normalised ({r['norm_lo']:.3f}..{r['norm_hi']:.3f})",
                None if r["ids_unique"] else "duplicate stimulus_ids"]))
            bad += 0 if ok else 1
            tag = Path(f).name if len(files) > 1 else ""
            print(f"{name:20s} {tag:36s} {r['n_img']:5d} x {r['dim']:5d}  "
                  f"norm {r['norm_lo']:.4f}..{r['norm_hi']:.4f}  "
                  f"{r.get('epoch',''):18s} {'PASS' if ok else 'FAIL'}{why}")
    print("\n" + ("ALL BACKBONES PASS" if bad == 0 else f"{bad} file(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
