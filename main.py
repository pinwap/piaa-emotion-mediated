"""Single entry point for the project.

Examples:
    uv run main.py verify --splits
    uv run main.py table1
    uv run main.py backbone --backbones clip,clip_ft,clip_ft_emo,qwen4b,qwen8b
    uv run main.py efficiency --n-train 10,25,50,100
    uv run main.py faithfulness

Every command writes to output/<experiment name>/, always with a config.json next to it.
"""
from __future__ import annotations

import os
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)

# *** bit-for-bit reproducibility -- must run before numpy is imported ***
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config                                    # noqa: E402
from src.data.data import XpassDataset                           # noqa: E402
from src.data.splits import V4Split                              # noqa: E402
from src.modeling.backbones import get_backbone                  # noqa: E402
from src.modeling.pipeline import Pipeline                       # noqa: E402


def build(cfg: Config, backbone_name: str | None = None):
    """Wire up dataset + backbone + split + pipeline from config."""
    ds = XpassDataset(cfg.data_dir, first_session_only=cfg.first_session_only)
    bb = get_backbone(backbone_name or cfg.backbone, cfg.features_dir)
    sp = V4Split(cfg.split_dir, n_folds=cfg.n_folds)
    return ds, bb, sp, Pipeline(cfg, ds, bb, sp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment",
                    choices=["table1", "backbone", "efficiency", "faithfulness",
                            "formulas",
                            "stage1_emotion_acc",
                            "stage2_emotion_importance", "gt_formula_matched",
                            "fig_faithfulness",
                            "fig_efficiency", "verify"])
    ap.add_argument("--backbone", default="qwen8b", help="default: qwen8b")
    ap.add_argument("--backbones", default="clip,clip_ft,clip_ft_emo,qwen4b,qwen8b")
    ap.add_argument("--n-train", default="10,25,50,100")
    ap.add_argument("--all-sessions", action="store_true",
                    help="disable the first-session filter (not what the paper uses)")
    ap.add_argument("--splits", action="store_true", help="verify: check the data split")
    ap.add_argument("--repro", metavar="EXPERIMENT", default=None,
                    help="verify: run EXPERIMENT twice, check the results are "
                         "byte-identical (e.g. --repro efficiency)")
    ap.add_argument("--seed", default=None,
                    help="table1: run only these seeds, e.g. --seed 1 "
                         "(seeds are independent and may run in parallel; "
                         "omit --seed to run and merge all three)")
    ap.add_argument("--stage2", default=None, choices=["plain", "A", "B", "C"],
                    help="table1: how the personal head relates to the population model. "
                         "plain=shrink toward 0, "
                         "A=GIAA prediction as an extra feature, "
                         "B=shrink toward the pooled training-group weights, "
                         "C=fit on the residual against GIAA (the paper's "
                         "population-anchored model). Each writes its "
                         "own summary{_A,_B,_C}.csv so they can be compared.")
    ap.add_argument("--mediators", default=None,
                    help="table1/efficiency: comma-separated mediators, e.g. "
                         "--mediators emotion,emotion_sd,emotion_hist. "
                         "emotion_sd (14 wide) and emotion_hist (35 wide) are "
                         "the distribution-valued Stage-1 variants.")
    ap.add_argument("--heads", default=None,
                    help="table1: comma-separated heads to run. A partial run writes "
                         "per_unit*_<h>_seed*.csv and leaves the full run's "
                         "files untouched, then writes its own merged summary")
    ap.add_argument("--folds", default=None,
                    help="efficiency: run only these fold indices, e.g. "
                         "--folds 0,1. Folds are independent, so this is how "
                         "one seed is spread over cores; merge the resulting "
                         "raw*.csv files with src/utils/tools/ingest_runs.py")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--joint-concept-weight", type=float, default=None,
                    help="lambda in L=MSE(score) + lambda*MSE(concept), "
                         "as in Koh et al. (2020). Default 1.0.")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.all_sessions:
        cfg.first_session_only = False
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    cfg.backbone = args.backbone
    if args.stage2 is not None:
        # Set this before ``build`` so the run's config.json records the same
        # variant that is passed to the experiment.
        cfg.stage2_variant = args.stage2
    if args.joint_concept_weight is not None:
        cfg.joint_concept_weight = args.joint_concept_weight

    if args.experiment == "verify":
        from src.utils import verify
        if args.repro:
            ds, bb, sp, pipe = build(cfg, args.backbone)
            name = args.repro
            if name == "efficiency":
                from src.experiments import efficiency
                go = lambda: efficiency.run(cfg, pipe, [int(x) for x in args.n_train.split(",")])  # noqa: E731
            elif name == "faithfulness":
                from src.experiments import faithfulness
                go = lambda: faithfulness.run(cfg, pipe)                      # noqa: E731
            elif name == "stage1_emotion_acc":
                from src.experiments import stage1_emotion_acc
                go = lambda: stage1_emotion_acc.run(cfg, pipe)                # noqa: E731
            elif name == "stage2_emotion_importance":
                from src.experiments import stage2_emotion_importance
                go = lambda: stage2_emotion_importance.run(cfg, pipe)         # noqa: E731
            else:
                raise SystemExit(f"--repro does not know experiment '{name}'")
            return 0 if verify.check_repro(cfg, name, go) else 1
        return 0 if verify.check_splits(cfg) else 1

    # Stored-result analyses -> able to regenerate diagnostics and figures from CSV databases.
    if args.experiment == "faithfulness":
        from src.experiments import faithfulness
        faithfulness.run(cfg)
        return 0
    if args.experiment == "fig_faithfulness":
        from src.utils import fig_faithfulness
        fig_faithfulness.run(cfg, "sem")
        return 0
    if args.experiment == "fig_efficiency":
        from src.utils import fig_efficiency
        fig_efficiency.run(cfg, "sem")
        return 0

    ds, bb, sp, pipe = build(cfg, args.backbone)

    # --every experiment--
    if args.experiment == "table1":
        from src.experiments import table1
        cfg.n_train = int(args.n_train.split(",")[-1])
        heads = args.heads.split(",") if args.heads else None
        meds = args.mediators.split(",") if args.mediators else None
        if args.seed is not None:
            for s in (int(x) for x in str(args.seed).split(",")):
                table1.run_one_seed(cfg, pipe, s, args.stage2, heads, meds)
        elif heads is not None or meds is not None:
            for s in table1.SEEDS:
                table1.run_one_seed(cfg, pipe, s, args.stage2, heads, meds)
            table1.merge_partial_runs(
                cfg, heads or table1.HEADS, meds or table1.MEDIATORS,
                args.stage2)
        else:
            table1.run(cfg, pipe, ds, variant=args.stage2)
    elif args.experiment == "backbone":
        from src.experiments import backbone
        backbone.run(cfg, args.backbones.split(","), variant=args.stage2,
                     heads=args.heads.split(",") if args.heads else None)
    elif args.experiment == "efficiency":
        from src.experiments import efficiency
        efficiency.run(cfg, pipe, [int(x) for x in args.n_train.split(",")],
                       variant=args.stage2,
                       heads=args.heads.split(",") if args.heads else None,
                       mediators=args.mediators.split(",") if args.mediators else None,
                       backbone=args.backbone,
                       seeds=([int(x) for x in str(args.seed).split(",")]
                              if args.seed is not None else (0,)),
                       folds=([int(x) for x in args.folds.split(",")]
                              if args.folds else None))
    elif args.experiment == "formulas":
        from src.experiments import formulas
        formulas.run(
            cfg, pipe, variant=args.stage2 or "C",
            n_train=int(args.n_train.split(",")[-1]),
            mediator=(args.mediators.split(",")[0]
                      if args.mediators else "emotion"),
            seed=int(str(args.seed).split(",")[0]) if args.seed else 0)
    elif args.experiment == "stage1_emotion_acc":
        from src.experiments import stage1_emotion_acc
        stage1_emotion_acc.run(cfg, pipe)
    elif args.experiment == "stage2_emotion_importance":
        from src.experiments import stage2_emotion_importance
        stage2_emotion_importance.run(cfg, pipe)
    elif args.experiment == "gt_formula_matched":
        from src.experiments import gt_formula_matched
        gt_formula_matched.run(cfg, pipe, n_train=int(args.n_train.split(",")[-1]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
