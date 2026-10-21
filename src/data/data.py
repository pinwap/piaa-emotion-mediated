"""Load XPASS-Vis ratings and apply the first-occurrence protocol.

For the 4,509 repeated user--image pairs, model fitting uses the first
occurrence. Both occurrences remain available through :meth:`retest_pairs`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EMOTION_COLS = ["like", "beautiful", "impressed", "intellectual", "motivated",
                "amused", "nostalgic", "sad", "distasteful"]
CORE7 = [c for c in EMOTION_COLS if c not in ("like", "beautiful")]
COLUMN_MAP = {
    "user_id": "user_id",
    "stimulus_id": "sample_id",
    "domain": "genre",
    "overall": "Aesthetic",
    "like": "Like",
    "beautiful": "Beautiful",
    "distasteful": "Distasteful",
    "impressed": "Impressed",
    "intellectual": "Intellectually",
    "motivated": "Motivated",
    "nostalgic": "Nostalgic",
    "sad": "Sad",
    "amused": "Amused",
}
DOMAIN_NORMALIZE = {"art": "art", "fashion": "fashion",
                    "landscape": "landscape", "scenery": "landscape"}
DOMAINS = ["art", "fashion", "landscape"]
DOMAIN_LABEL = {"art": "Artwork", "fashion": "Fashion", "landscape": "Landscape"}

EXPECTED = {"n_interactions": 87836, "n_users": 129, "n_stimuli": 6526}

def load_raw(data_dir: str | Path) -> pd.DataFrame:
    # Read ratings.csv
    data_dir = Path(data_dir)
    files_path = data_dir / "ratings.csv"
    if not files_path.exists():
        raise FileNotFoundError(f"no ratings.csv found in {data_dir}")
    
    df = pd.read_csv(files_path)
    df = df.rename(columns={v: k for k, v in COLUMN_MAP.items() if v in df.columns})

    need = ["user_id", "stimulus_id", "domain", "overall"]+CORE7
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing} (have: {list(df.columns)})")

    # Emotion items are stored as 0--4 in the released file; expose them on
    # the questionnaire's 1--5 scale.  The overall aesthetic score remains on
    # its 0--6 file coding internally; adding a constant would not change the
    # correlation metrics used by the paper.
    for col in EMOTION_COLS:
        if col in df.columns:
            df[col] = df[col] + 1

    df["domain"] = df["domain"].map(DOMAIN_NORMALIZE)
    if df["domain"].isna().any():
        raise ValueError(f"unrecognized domain value: {df.loc[df.domain.isna(), 'domain'].unique()}")
    return df


class XpassDataset:
    """
    Usage:
        ds = XpassDataset(cfg.data_dir, first_session_only=True)
        sub = ds.subset(domain="art", users=fold.train_users)
        agg = ds.per_stimulus(sub)
    """

    def __init__(self, data_dir: str | Path, first_session_only: bool = True, verbose: bool = True):
        raw = load_raw(data_dir).dropna(subset=CORE7 + ["overall"])
        raw = raw.reset_index(drop=True)
        raw["_row"] = np.arange(len(raw))
        self.n_raw = len(raw)
        self.first_session_only = first_session_only

        if first_session_only:
            df = raw.sort_values("_row")
            df = df.drop_duplicates(subset=["user_id", "stimulus_id"], keep="first")
            if verbose:
                print(f"[data] first-session only: {self.n_raw} -> {len(df)} rows, dropped {self.n_raw - len(df)} duplicates")
        else:
            df = raw
            if verbose:
                print(f"[data] using all ratings: {len(df)} rows")
        self.df = df

    def subset(self, domain: str | None = None, users=None) -> pd.DataFrame:
        df = self.df
        if domain is not None:
            df = df[df["domain"] == domain]
        if users is not None:
            df = df[df["user_id"].isin(set(users))]
        return df

    @staticmethod
    def per_stimulus(df: pd.DataFrame) -> pd.DataFrame:
        # Returns one row mean overall + CORE7 columns per stimulus(image), indexed by stimulus_id (str).
        return df.groupby(df["stimulus_id"].astype(str))[["overall"] + CORE7].mean()

    @staticmethod
    def per_stimulus_spread(df: pd.DataFrame) -> pd.DataFrame:
        """Per image, how much the raters disagreed about each emotion.

        The population mean throws this away: an image that leaves everyone
        mildly amused and one that splits the room average to the same number.
        Returned as the per-emotion standard deviation across raters, indexed
        and ordered exactly like per_stimulus() so the two can be concatenated
        column-wise.

        Images with one rater have undefined spread and are filled with zero;
        the released split has at least five raters per Stage-1 image.
        """
        return df.groupby(df["stimulus_id"].astype(str))[CORE7].std(ddof=0).fillna(0.0)

    @staticmethod
    def per_stimulus_hist(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """Per image, the full rating distribution of each emotion.

        Columns are <emotion>_b1..b{n_bins}: the share of raters who gave that
        emotion that rating, so each emotion's bins sum to 1. The scale is the
        integers 1..n_bins. Same index and order as per_stimulus().

        Each bin is built as a 0/1 indicator per rating and then averaged over
        the raters of an image, since the mean of an indicator is the share.
        """
        columns = {}
        for emotion in CORE7:
            ratings = df[emotion].to_numpy(float)
            for level in range(1, n_bins + 1):
                columns[f"{emotion}_b{level}"] = (ratings == level).astype(float)

        wide = pd.DataFrame(columns, index=df.index)
        return wide.groupby(df["stimulus_id"].astype(str)).mean()

    def population_emotions(self, d: pd.DataFrame) -> pd.DataFrame:
        # Population-mean emotion ratings per image (used to fit the shared mediator).
        return self.per_stimulus(d)

    def user_ids(self, domain: str | None = None):
        return sorted(self.subset(domain=domain)["user_id"].unique())

    def restrict_to_features(self, feature_ids) -> None:
        # Drop rows that image has no feature vector
        ids = [str(i) for i in feature_ids]
        self.df = self.df[self.df["stimulus_id"].astype(str).isin(ids)]


    # test-retest
    def retest_pairs(self, data_dir: str | Path) -> pd.DataFrame:
        # (first, second) rating pairs for images a user rated twice
        raw = load_raw(data_dir).dropna(subset=CORE7 + ["overall"]).reset_index(drop=True)
        
        raw["occ"] = raw.groupby(["user_id", "stimulus_id"]).cumcount()
        first = raw[raw["occ"] == 0]
        second = raw[raw["occ"] == 1]
        
        key = ["user_id", "stimulus_id", "domain"]
        return first.merge(second, on=key, suffixes=("_r1", "_r2"))

    def single_occurrence(self, data_dir: str | Path) -> pd.DataFrame:
        # Rows for images a user rated exactly once (used to fit the ceiling-analysis formula)
        raw = load_raw(data_dir).dropna(subset=CORE7 + ["overall"])
        count = raw.groupby(["user_id", "stimulus_id"]).size().rename("n_occ")
        joined = raw.join(count, on=["user_id", "stimulus_id"])
        return joined[joined["n_occ"] == 1]
