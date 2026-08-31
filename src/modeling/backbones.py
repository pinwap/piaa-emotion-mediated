"""Backbone = turns an image into a feature vector (frozen, not trained here).

Features are pre-extracted to .npz (see src/data/extract_*.py if you need
to redo it). Classes here just load and index by stimulus_id.

Fine-tuned CLIP features are loaded from a separate file for each fold; each
file is fitted using that fold's training users only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")
    z = np.load(path, allow_pickle=True)
    return {str(s): f for s, f in zip(z["stimulus_ids"], z["features"])}


class Backbone(ABC):
    """Common interface for every backbone."""

    name: str = "backbone"

    @abstractmethod
    def features_for_fold(self, fold_index: int) -> dict[str, np.ndarray]:
        """Return stimulus_id(str) -> feature vector for this fold."""

    def matrix(self, feats: dict[str, np.ndarray], ids) -> np.ndarray:
        """Stack features in the order of `ids`."""
        return np.stack([feats[str(s)] for s in ids])


class FrozenBackbone(Backbone):
    """One feature file, same for every fold (never trained on our data, no leak)."""

    def __init__(self, path: str | Path, name: str):
        self.path = Path(path)
        self.name = name
        self._cache: dict[str, np.ndarray] | None = None

    def features_for_fold(self, fold_index: int) -> dict[str, np.ndarray]:
        if self._cache is None:
            self._cache = _load_npz(self.path)
        return self._cache


class PerFoldBackbone(Backbone):
    """Features fine-tuned separately per fold -- load a different file per fold.

    `pattern` takes {fold} as a placeholder, e.g.
    "clip_ftpf_overall_v4_fold{fold}.npz".
    """

    def __init__(self, features_dir: str | Path, pattern: str, name: str):
        self.dir = Path(features_dir)
        self.pattern = pattern
        self.name = name
        self._cache: dict[int, dict[str, np.ndarray]] = {}

    def features_for_fold(self, fold_index: int) -> dict[str, np.ndarray]:
        if fold_index not in self._cache:
            self._cache = {}   # retain only the active fold in memory
            self._cache[fold_index] = _load_npz(
                self.dir / self.pattern.format(fold=fold_index))
        return self._cache[fold_index]


#: backbone registry -- add new ones here, nowhere else
BACKBONE_SPECS = {
    "clip":     dict(kind="frozen",  file="clip_features.npz",  label="CLIP frozen"),
    "clip_ft":  dict(kind="perfold",
                     pattern="clip_ftpf_overall_v4_results/"
                             "clip_ftpf_overall_v4_fold{fold}.npz",
                     label="CLIP-ft (score)"),
    # fine-tuned to predict the 7 emotions rather than the score, so the
    # backbone is aligned to the bottleneck instead of to the target
    "clip_ft_emo": dict(kind="perfold",
                        pattern="clip_ftpf_emotion_v4_results/"
                                "clip_ftpf_emotion_v4_fold{fold}.npz",
                        label="CLIP-ft (emotion)"),
    "qwen4b":   dict(kind="frozen",  file="vlm4b_LT15.npz",     label="Qwen3-VL 4B"),
    "qwen8b":   dict(kind="frozen",  file="vlm_LT15.npz",       label="Qwen3-VL 8B"),
}


def get_backbone(name: str, features_dir: str | Path) -> Backbone:
    if name not in BACKBONE_SPECS:
        raise KeyError(f"unknown backbone '{name}' (have: {list(BACKBONE_SPECS)})")
    spec = BACKBONE_SPECS[name]
    if spec["kind"] == "frozen":
        return FrozenBackbone(Path(features_dir) / spec["file"], spec["label"])
    return PerFoldBackbone(features_dir, spec["pattern"], spec["label"])


def backbone_label(name: str) -> str:
    return BACKBONE_SPECS[name]["label"]
