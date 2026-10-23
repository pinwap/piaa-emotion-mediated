# ================================================================
# create 5-fold leakage-free (user + image disjoint) data split
# ================================================================
import json
import sys
from pathlib import Path

import pandas as pd
from src.config import PROJECT_ROOT

DATA = PROJECT_ROOT / "Dataset"
OUTDIR = DATA / "split_v4_10group"

# fold -> (test groups, val group) ; train = the rest
FOLDS = [
    ([0, 1], 2),
    ([2, 3], 4),
    ([4, 5], 6),
    ([6, 7], 8),
    ([8, 9], 0),
]
ALL_GROUPS = set(range(10))


def main():
    df = pd.read_csv(DATA / "maked" / "ratings.csv")
    # map group (set) -> users, images
    grp_users = {s: set(df[df["set"] == s].user_id.unique()) for s in ALL_GROUPS}
    grp_imgs = {s: set(df[df["set"] == s].sample_id.unique()) for s in ALL_GROUPS}

    # sanity: user/image disjoint across group 
    assert (df.groupby("user_id")["set"].nunique() == 1).all(), "user crosses group!"
    assert (df.groupby("sample_id")["set"].nunique() == 1).all(), "image crosses group!"

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{'fold':>4} | {'test grp':>9} | val | {'train grp':>15} | {'#test_u':>7} {'#val_u':>6} {'#train_u':>8} | {'#GIAA_img':>9}")
    print("-" * 90)

    summary = []
    for fi, (test_g, val_g) in enumerate(FOLDS):
        train_g = sorted(ALL_GROUPS - set(test_g) - {val_g})
        
        test_u = sorted(set().union(*[grp_users[s] for s in test_g]))
        val_u = sorted(grp_users[val_g])
        train_u = sorted(set().union(*[grp_users[s] for s in train_g]))
        
        giaa_img = sorted(set().union(*[grp_imgs[s] for s in train_g]))
        test_img = set().union(*[grp_imgs[s] for s in test_g])
        val_img = grp_imgs[val_g]

        # ── verify disjoint ──
        assert not (set(test_u) & set(val_u)), f"fold{fi}: test & val users!"
        assert not (set(test_u) & set(train_u)), f"fold{fi}: test & train users!"
        assert not (set(val_u) & set(train_u)), f"fold{fi}: val & train users!"
        assert not (set(giaa_img) & test_img), f"fold{fi}: GIAA & test images!"
        assert not (set(giaa_img) & val_img), f"fold{fi}: GIAA & val images!"

        fd = OUTDIR / f"fold{fi}"
        fd.mkdir(exist_ok=True)
        (fd / "train_users.txt").write_text("\n".join(map(str, train_u)), encoding="utf-8")
        (fd / "val_users.txt").write_text("\n".join(map(str, val_u)), encoding="utf-8")
        (fd / "test_users.txt").write_text("\n".join(map(str, test_u)), encoding="utf-8")
        (fd / "giaa_train_images.txt").write_text("\n".join(map(str, giaa_img)), encoding="utf-8")
        (fd / "meta.json").write_text(json.dumps(
            {"test_groups": test_g, "val_group": val_g, "train_groups": train_g,
             "n_test_users": len(test_u), "n_val_users": len(val_u),
             "n_train_users": len(train_u), "n_giaa_images": len(giaa_img)},
            indent=2), encoding="utf-8")

        print(f"{fi:>4} | {str(test_g):>9} | {val_g:>3} | {str(train_g):>15} | "
              f"{len(test_u):>7} {len(val_u):>6} {len(train_u):>8} | {len(giaa_img):>9}")
        summary.append((len(test_u), len(val_u), len(train_u)))

    # ── verify coverage ──
    all_users = set(df.user_id.unique())
    test_counts = pd.Series(0, index=sorted(all_users))
    for fi, (test_g, _) in enumerate(FOLDS):
        for s in test_g:
            for u in grp_users[s]:
                test_counts[u] += 1
    print("-" * 90)
    print(f"coverage: each user is in test set {sorted(test_counts.unique())} times (should be [1]) | total users = {len(all_users)}")
    print(f"\nAll disjoint checks passed (user + image are separated completely in each fold)")
    print(f"Results saved to: {OUTDIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
