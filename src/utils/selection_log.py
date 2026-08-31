"""Record validation scores for every hyperparameter candidate.

Each selection stores the full candidate grid, scores, selected row, tie size,
margin, and boundary status. ``Pipeline._eval_fold_domain`` may run in a
joblib worker, so ``collecting`` uses context-local buffers that are returned
with the fold-domain results. Calls outside a ``collecting`` block are no-ops.
"""
from __future__ import annotations

import contextvars
from pathlib import Path

import numpy as np
import pandas as pd

#: context of the run, stamped onto every row. Held in a ContextVar rather
#: than a plain global so a threaded caller cannot mix two folds together.
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("selection_log_active",
                                                         default=None)
_CONTEXT: contextvars.ContextVar = contextvars.ContextVar("selection_log_context",
                                                          default={})

#: what identifies the run this selection happened in
CONTEXT = ["experiment", "backbone", "variant", "n_train", "seed", "fold", "domain"]
#: what was being selected. kind separates the two sorts of row this file
#: holds: "selection" rows are candidates that were ranked against each
#: other, "diagnostic" rows are single measured facts about the model that
#: won (epochs run, support size). Mixing them without a flag would let a
#: mean over `score` average a learning rate against an epoch count.
SUBJECT = ["kind", "stage", "component", "mediator", "head"]
#: the candidate itself -- one column per hyperparameter the project
#: selects. A ridge candidate fills `alpha` only, an MLP candidate fills `lr`
#: and `epochs`; the columns a candidate does not use stay NaN so a reader
#: never has to know which head wrote the row to read the number correctly.
CANDIDATE = ["lr", "alpha", "epochs"]
#: how it was ranked and what came of it. n_val_scored is how many things the
#: criterion averaged over and n_val_kind says what they were: user-units for
#: the personal head (it averages SROCC per validation user), images for
#: everything selected on a group-level MSE.
OUTCOME = ["criterion", "score", "lower_is_better", "selected", "rank",
           "n_candidates", "tie_size", "edge", "at_grid_edge", "margin",
           "n_val_scored", "n_val_kind", "n_val_users", "note"]

COLUMNS = CONTEXT + SUBJECT + CANDIDATE + OUTCOME


class collecting:
    """Open a buffer for `note()` and hand it back on exit.

    Used as a context manager::

        with selection_log.collecting(fold=0, domain="art", ...) as records:
            ...                       # anything that selects appends here
        return rows, records
    """

    def __init__(self, **context):
        self.context = context
        self.records: list[dict] = []
        self._tokens = ()

    def __enter__(self) -> list[dict]:
        self._tokens = (_ACTIVE.set(self.records),
                        _CONTEXT.set(dict(self.context)))
        return self.records

    def __exit__(self, *exc):
        a, c = self._tokens
        _ACTIVE.reset(a)
        _CONTEXT.reset(c)
        return False


def active() -> bool:
    """True inside a `collecting()` block -- lets a caller skip the work of
    assembling a grid nobody is going to read."""
    return _ACTIVE.get() is not None


def note(stage: str, component: str, criterion: str,
         candidates, scores, chosen, *,
         lower_is_better: bool, axis2: str = "alpha",
         mediator=None, head=None,
         n_val_scored=None, n_val_kind=None, n_val_users=None,
         extra_note=None) -> None:
    """Record one selection: the whole grid, ranked, with the winner flagged.

    candidates  the grid, in the order it was scored. Each entry is either a
                scalar (a ridge penalty, logged as lr=NaN) or an (lr, alpha)
                pair.
    scores      the value each candidate was ranked by, same order.
    chosen      the candidate that won, in the same form as `candidates`.
    lower_is_better  True for an MSE or a loss, False for a correlation.
    """
    buf = _ACTIVE.get()
    if buf is None:
        return

    cands = [_split(c) for c in candidates]
    scores = np.asarray(list(scores), float)
    won = _split(chosen)
    best = float(np.nanmin(scores) if lower_is_better else np.nanmax(scores))
    # rank 1 = the candidate this criterion liked most
    order = np.argsort(scores if lower_is_better else -scores, kind="stable")
    rank = np.empty(len(scores), int)
    rank[order] = np.arange(1, len(scores) + 1)

    # a tie is what the selector actually treated as one: see ALPHA_TIE_RTOL
    from src.modeling.heads import ALPHA_TIE_RTOL
    if lower_is_better:
        tied = scores <= best * (1.0 + ALPHA_TIE_RTOL)
    else:
        tied = scores >= best - abs(best) * ALPHA_TIE_RTOL
    tie_size = int(tied.sum())

    # margin = how much better the winner was than the best candidate that is
    # not tied with it. Small margin -> the choice is noise, whatever it says.
    rest = scores[~tied]
    margin = (float(abs(best - (rest.min() if lower_is_better else rest.max())))
              if rest.size else float("nan"))

    edge = _edge(won, cands, axis2)
    ctx = dict(_CONTEXT.get())
    for (lr, second), s, r, t in zip(cands, scores, rank, tied):
        # the second axis is a weight decay for one head and an epoch budget
        # for another, so it gets its own column either way and the unused
        # one stays NaN. Reading `alpha` as epochs would misreport the log.
        alpha = second if axis2 == "alpha" else float("nan")
        epochs = second if axis2 == "epochs" else float("nan")
        buf.append({
            **{k: ctx.get(k) for k in CONTEXT},
            "kind": "selection",
            "stage": stage, "component": component,
            "mediator": mediator, "head": head,
            "lr": lr, "alpha": alpha, "epochs": epochs,
            "criterion": criterion, "score": float(s),
            "lower_is_better": bool(lower_is_better),
            "selected": int(_same((lr, second), won)),
            "rank": int(r),
            "n_candidates": len(cands),
            "tie_size": tie_size,
            "edge": edge,
            "at_grid_edge": int(bool(edge)),
            "margin": margin,
            "n_val_scored": n_val_scored,
            "n_val_kind": n_val_kind,
            "n_val_users": n_val_users,
            "note": extra_note,
        })


def note_fit(stage: str, component: str, criterion: str, value, *,
             mediator=None, head=None, lr=None, alpha=None, epochs=None,
             extra_note=None) -> None:
    """Record a single measured fact about a fitted model rather than a
    selection -- how many epochs the winner actually ran, how many test users
    hit `mlp_max_iter`, how wide the mediator was. Same table, one row, no
    grid: `criterion` names the quantity and `score` carries it."""
    buf = _ACTIVE.get()
    if buf is None:
        return
    ctx = dict(_CONTEXT.get())
    buf.append({
        **{k: ctx.get(k) for k in CONTEXT},
        "kind": "diagnostic",
        "stage": stage, "component": component,
        "mediator": mediator, "head": head,
        "lr": lr, "alpha": alpha, "epochs": epochs,
        "criterion": criterion,
        "score": float(value) if value is not None else float("nan"),
        "lower_is_better": None, "selected": 1, "rank": 1,
        "n_candidates": 1, "tie_size": 1, "edge": "", "at_grid_edge": 0,
        "margin": float("nan"), "n_val_scored": None, "n_val_kind": None,
        "n_val_users": None, "note": extra_note,
    })


def _split(c):
    """(lr, alpha) for an MLP candidate, (nan, penalty) for a ridge one."""
    if isinstance(c, (tuple, list, np.ndarray)):
        lr, alpha = c
        return (float(lr), float(alpha))
    return (float("nan"), float(c))


def _same(a, b) -> bool:
    """Candidate equality that treats NaN as a value. A ridge candidate is
    (NaN, penalty), and `NaN == NaN` is False, so a plain tuple comparison
    marks every ridge selection unselected."""
    return all((np.isnan(x) and np.isnan(y)) or x == y for x, y in zip(a, b))


def _edge(won, cands, axis2: str = "alpha") -> str:
    """Which axis the winner ran out of grid on, and at which end --
    "alpha:max" means the strongest weight decay on offer won, so the grid
    stopped before the optimum did and the number is a limit, not a choice.

    A bare boolean is useless here: the MLP grid has three values per axis,
    so all but the middle one is an edge and the flag would be on almost
    always. The end is what carries the warning.
    """
    out = []
    for i, axis in ((0, "lr"), (1, axis2)):
        vals = sorted({c[i] for c in cands if not np.isnan(c[i])})
        if len(vals) < 2 or np.isnan(won[i]):
            continue
        if won[i] == vals[0]:
            out.append(f"{axis}:min")
        elif won[i] == vals[-1]:
            out.append(f"{axis}:max")
    return ",".join(out)


def to_frame(records) -> pd.DataFrame:
    """Accepts the raw list of dicts or an already-built frame -- `pd.DataFrame`
    of a DataFrame would otherwise silently build a frame of its column
    names."""
    d = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if d.empty:
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in d.columns:
            d[col] = pd.NA
    return d[COLUMNS]


def write(records, path: Path | str) -> Path:
    """Write the complete selection log for one run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(records).to_csv(path, index=False)
    return path


def summarize(records, kind: str = "selection") -> pd.DataFrame:
    """Return one selected row per search; pass ``kind=None`` for diagnostics."""
    d = to_frame(records)
    if d.empty:
        return d
    d = d[d["selected"] == 1]
    if kind is not None:
        d = d[d["kind"] == kind]
    return d.reset_index(drop=True)
