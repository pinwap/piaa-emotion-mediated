"""Suite: 'thundering_lobster'.

Qwen3-VL 8B MLP Series on pure-MSE training (Anchor C, n=100, MLP head).

Re-runs the whole MLP series after two things changed underneath the results
already on disk:

  * weight decay stopped being a searched axis and is fixed at 0.0, so every
    MLP here -- the sequential extractor, the joint extractor and the personal
    head -- trains on plain MSE with no L2 term;
  * the learning-rate grid was widened to eleven half-decade values from 1e-4
    to 1e1, because on the previous grid the selector hit the boundary at both
    ends and a winner on the edge reports where the search stopped rather than
    where the optimum is.

All seven mediators run under one head in one command, so Table 1 is
internally comparable: the two MLP Stage-1 rows sit beside the same controls
they have to beat, selected by the same protocol, on the same units.
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

#: every mediator, in one command. They are not split across steps because
#: Pipeline.shared_context fits them together per (fold, domain) -- one step
#: per mediator would refit Stage-1 seven times over for the same numbers.
MEDIATORS = ("identity,random,shuffled,pca,emotion,"
             "emotion_mlp,emotion_joint")


def _step(step_id: int, seed: int) -> SuiteStep:
    """One seed. Seeds are independent, which is what makes them the axis
    worth flattening on: any one can be re-run alone after a failure."""
    return SuiteStep(
        id=step_id,
        codename=f"{step_id}_mlp-mse-s{seed}",
        folder=f"{step_id}_mlp-mse-s{seed}",
        title=f"MLP Series, pure MSE, all 7 mediators (Seed {seed}, n=100, Anchor C)",
        desc=(f"Controls (identity, random, shuffled, pca), ridge Stage-1 "
              f"(emotion) and both MLP Stage-1 rows (emotion_mlp sequential, "
              f"emotion_joint) under the MLP personal head on Qwen3-VL 8B at "
              f"n=100, seed {seed}, Anchor C. No weight decay; learning rate "
              f"selected on the validation group over 1e-4..1e1."),
        cmd=[
            sys.executable, "main.py", "efficiency",
            "--backbone", "qwen8b",
            "--mediators", MEDIATORS,
            "--heads", "mlp",
            "--n-train", "100",
            "--seed", str(seed),
            "--stage2", "C",
        ],
    )


SUITE = Suite(
    name="thundering_lobster",
    title="Qwen3-VL 8B MLP Series, Pure MSE, Wide LR Grid (Anchor C, n=100, MLP Head)",
    desc=("Rebuilds the MLP series with no L2 penalty (mlp_alpha=0) and an "
          "eleven-value learning-rate grid from 1e-4 to 1e1, across all seven "
          "mediators under the MLP head on Qwen3-VL 8B at n=100, Anchor C, "
          "seeds 0, 1, 2."),
    steps=[_step(i, seed) for i, seed in enumerate((0, 1, 2), start=1)],
)
