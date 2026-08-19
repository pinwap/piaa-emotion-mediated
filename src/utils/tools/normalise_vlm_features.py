"""Bring a saved VLM feature file into line with Ryu & Yanaka.

The reference paper L2-normalises the pooled hidden state, and
select_vlm_layer.py does that by default (--no_norm is opt-in). One file
in features/ was nevertheless written without it: vlm_LT15.npz (Qwen3-VL 8B)
had row norms around 758-765, while vlm4b_LT15.npz (Qwen3-VL 4B) had norms
of exactly 1. That made the two Qwen rows differ in preprocessing as well as
in model size, so any gap between them was uninterpretable.

Re-extraction is not needed: L2-normalisation is a per-row rescale of the
saved matrix, so applying it here gives exactly what select_vlm_layer.py
would have written without --no_norm.

    uv run python tools/normalise_vlm_features.py features/vlm_LT15.npz

Writes <name>_backup.npz first and refuses to overwrite an existing backup,
so re-running cannot destroy the original. A file that is already normalised
is left alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOL = 1e-3


def norms_of(path: Path) -> np.ndarray:
    return np.linalg.norm(np.load(path)["features"], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="a *_LT*.npz feature file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.path
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    z = np.load(src, allow_pickle=True)
    missing = {"stimulus_ids", "features"} - set(z.files)
    if missing:
        raise SystemExit(f"{src.name} is missing keys: {sorted(missing)}")

    ids, feats = z["stimulus_ids"], z["features"].astype(np.float32)
    n = np.linalg.norm(feats, axis=1)
    print(f"{src.name}: {feats.shape}, row norms {n.min():.4f} .. {n.max():.4f}")

    if abs(n.mean() - 1.0) < TOL and n.std() < TOL:
        print("already L2-normalised; nothing to do")
        return 0

    backup = src.with_name(src.stem + "_backup.npz")
    if backup.exists():
        raise SystemExit(f"backup already exists, refusing to overwrite: {backup.name}\n"
                         f"delete it by hand if you really mean to re-run this")

    # Same expression select_vlm_layer.py uses, so the result is identical to
    # what a fresh extraction without --no_norm would have produced.
    out = feats / np.clip(n[:, None], 1e-8, None)

    if args.dry_run:
        m = np.linalg.norm(out, axis=1)
        print(f"[dry-run] would write {src.name}; new norms "
              f"{m.min():.6f} .. {m.max():.6f}; backup -> {backup.name}")
        return 0

    # Carry every other array through untouched. Newer fine-tune files also
    # hold history/best_epoch, and rewriting only ids+features would silently
    # drop the record of how the model was trained.
    extra = {k: z[k] for k in z.files if k not in ("stimulus_ids", "features")}
    if extra:
        print("preserving extra keys:", sorted(extra))

    np.savez_compressed(backup, stimulus_ids=ids, features=feats, **extra)
    print(f"backed up original -> {backup.name}")

    np.savez_compressed(src, stimulus_ids=ids, features=out, **extra)

    # Verify by reading back from disk, not from the array in memory.
    m = norms_of(src)
    b = np.load(backup)
    assert abs(m.mean() - 1.0) < 1e-6 and m.std() < 1e-6, "not normalised after write"
    assert np.array_equal(np.load(src)["stimulus_ids"], ids), "stimulus_ids changed"
    assert set(np.load(src, allow_pickle=True).files) == set(z.files), "keys lost"
    assert b["features"].shape == feats.shape, "backup shape mismatch"
    # direction must be untouched: normalising is a rescale, not a rotation
    cos = (np.load(src)["features"] * b["features"]).sum(1) / (
        np.linalg.norm(np.load(src)["features"], axis=1) * np.linalg.norm(b["features"], axis=1))
    assert cos.min() > 1 - 1e-5, f"direction changed (min cos {cos.min()})"

    print(f"wrote {src.name}: norms now {m.min():.6f} .. {m.max():.6f}")
    print(f"verified: ids unchanged, directions unchanged (min cos {cos.min():.8f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
