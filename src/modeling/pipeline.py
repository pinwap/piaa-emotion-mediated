"""Pipeline -- wires Backbone + Mediator + Head together and runs the v4
evaluation protocol.

Per (fold, domain):
  1. pull train-user images -> features Xg, population-mean emotions Eg
  2. fit every mediator on (Xg, Eg), freeze
  3. for each test user: split support/eval, fit the head on support,
     score on the fixed eval set
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.data import CORE7, DOMAINS, XpassDataset
from src.data.splits import V4Split, per_user_split, user_rng
from src.modeling.heads import (ALPHA_HEADS, ALPHA_TIE_RTOL, head_grid,
                                make_head)
from src.modeling.mediators import build_shared_mediators
from src.utils import selection_log
from src.utils.metrics import evaluate, srocc


@dataclass
class UserUnit:
    """One evaluation unit = one user x one domain (387 total)."""
    fold: int
    domain: str
    user_id: int
    X_train: np.ndarray      # support image features
    X_eval: np.ndarray       # eval image features
    y_train: np.ndarray      # user's scores on support
    y_eval: np.ndarray
    E_train: np.ndarray      # user's true emotion ratings on support (upper bound)
    E_eval: np.ndarray
    # stimulus ids behind X_train / X_eval, in the same row order. Only
    # the qualitative-figure export reads them; every other consumer
    # ignores them, and they default to empty so nothing that builds a
    # UserUnit positionally has to change.
    train_ids: tuple = ()
    eval_ids: tuple = ()


class _WithPopFeature:
    """Variant A: wraps a mediator so its output gains the GIAA prediction as
    one extra feature. transform() still takes raw image features, so this
    drops into any place a mediator is used -- including Direct, which is the
    point: if only Hybrid got the population prior, a gain from the prior
    would be misread as a gain from the bottleneck."""

    def __init__(self, med, pop_head):
        self.med = med
        self.pop_head = pop_head

    def transform(self, X):
        base = np.asarray(self.med.transform(X), float)
        pop = np.asarray(self.pop_head.predict(X), float).reshape(-1, 1)
        return np.hstack([base, pop])


class _ResidualHead:
    """Variant C for a head with no weight vector -- in practice, the MLP.

    Fits the wrapped head on y_u - y_pop and adds y_pop back at predict time,
    so the personal model degrades onto the population model exactly as the
    anchored ridge does. Nothing about the wrapped head changes, so its
    hyperparameter is still selected on the validation group by the same
    code path as every other condition.

    `scaler` is the training-group scaler _PopAnchoredRidge also uses.
    Standardizing here, rather than letting the wrapped head fit its own
    scaler on each user's support set, is what leaves "anchor C + ridge" and
    "anchor C + MLP" differing in the regressor and nothing else. Without it
    the MLP row would additionally carry a per-user standardization that the
    ridge row beside it does not, and a gap between the two rows would no
    longer be a statement about the model family.
    """

    is_linear = False

    def __init__(self, head, pop_head, scaler=None):
        self.head, self.pop_head, self.scaler = head, pop_head, scaler

    def _pop(self, X_raw):
        return np.asarray(self.pop_head.predict(X_raw), float).ravel()

    def _z(self, M):
        M = np.asarray(M, float)
        return M if self.scaler is None else self.scaler.transform(M)

    def fit(self, M, y, X_raw=None, frozen_alpha=None, frozen_lr=None, **_):
        r = np.asarray(y, float).ravel() - self._pop(X_raw)
        Z = self._z(M)
        if frozen_lr is not None:
            self.head.fit(Z, r, frozen_lr=frozen_lr)
        elif frozen_alpha is not None:
            self.head.fit(Z, r, frozen_alpha=frozen_alpha)
        else:
            self.head.fit(Z, r)
        return self

    def predict(self, M, X_raw=None):
        return (self._pop(X_raw)
                + np.asarray(self.head.predict(self._z(M)), float).ravel())

    @property
    def effective_dof(self):
        return getattr(self.head, "effective_dof", float("nan"))


class _PopAnchoredRidge:
    """Variants B and C: a personal ridge that degrades onto the population
    model rather than onto a constant.

    B shrinks the weights toward w_pop, the Stage-2 coefficients fit on the
    pooled training group, implemented as a residual fit so no new solver is
    needed: w_u = w_pop + ridge(X_u, y_u - X_u w_pop). At full shrinkage the
    correction vanishes and w_u = w_pop exactly.

    C is the constrained case with w_pop pinned: the head is fit on
    y_u - y_pop and its prediction added back to y_pop.

    Standardization is fixed on the training group in both cases. Fitting a
    fresh scaler per user would express w_pop in each user's own units and
    the anchor would no longer mean anything.
    """

    is_linear = True

    def __init__(self, cfg, mode: str, scaler, w_pop, b_pop, pop_head=None,
                 kind: str = "ridge"):
        self.cfg, self.mode, self.kind = cfg, mode, kind
        self.scaler, self.w_pop, self.b_pop = scaler, w_pop, b_pop
        self.pop_head = pop_head
        self._delta = None
        self._b = 0.0
        self._Z_train = None
        self._alpha = np.nan

    def _resid_model(self, alpha):
        """The regressor that learns the personal correction. Same family as
        the head being anchored, so 'lasso anchored on GIAA' really is a lasso
        on the residual rather than a ridge wearing its name."""
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        if self.kind == "lasso":
            return Lasso(alpha=float(alpha), max_iter=self.cfg.sparse_max_iter)
        if self.kind == "elastic":
            return ElasticNet(alpha=float(alpha),
                              l1_ratio=self.cfg.elastic_l1_ratio,
                              max_iter=self.cfg.sparse_max_iter)
        return Ridge(alpha=float(alpha))

    def fit(self, M, y, X_raw=None, frozen_alpha=None, **_):
        Z = self.scaler.transform(np.asarray(M, float))
        y = np.asarray(y, float)

        if self.mode == "B":
            resid = y - (Z @ self.w_pop + self.b_pop)
        else:                                    # C: residual against GIAA
            resid = y - np.asarray(self.pop_head.predict(X_raw), float)

        if frozen_alpha is not None:
            alpha = float(frozen_alpha)
        else:
            # no frozen value (ad-hoc use): fall back to the strongest penalty
            # in the grid rather than silently choosing on this user's own data
            from src.modeling.heads import head_grid
            alpha = float(np.max(np.asarray(head_grid(self.kind, self.cfg), float)))
        r = self._resid_model(alpha).fit(Z, resid)
        self._alpha = alpha
        self._delta, self._b = r.coef_.ravel(), float(r.intercept_)
        self._Z_train = Z
        return self

    def predict(self, M, X_raw=None):
        Z = self.scaler.transform(np.asarray(M, float))
        if self.mode == "B":
            return Z @ (self.w_pop + self._delta) + self.b_pop + self._b
        return np.asarray(self.pop_head.predict(X_raw), float) + Z @ self._delta + self._b

    def weights(self):
        return (self.w_pop + self._delta).copy() if self.mode == "B" else self._delta.copy()

    def effective_dof(self):
        from src.utils.metrics import effective_dof
        return effective_dof(
            self._Z_train, self._alpha, already_standardized=True
        )


class Pipeline:
    def __init__(self, cfg, dataset: XpassDataset, backbone, split: V4Split):
        self.cfg = cfg
        self.ds = dataset
        self.backbone = backbone
        self.split = split
        #: every hyperparameter the last run_grid() selected, with the grid it
        #: was selected from. The caller writes it next to its results; see
        #: src/utils/selection_log.py for the schema.
        self.selection_records: list[dict] = []
        #: stamped onto those rows so one selection log can be read next to
        #: another. Set by whichever experiment is driving.
        self.experiment_name: str | None = None

    def iter_units(self, fold, domain: str, feats, n_train: int | None = None,
                  users=None):
        """Evaluation units for one fold/domain.

        users  defaults to the fold's test users. Passing fold.val_users
              instead runs the identical support/eval split procedure on the
              validation group, which is what lets a hyperparameter be
              selected by mirroring the test protocol rather than guessed at.
        """
        cfg = self.cfg
        n_train = n_train or cfg.n_train
        users = fold.test_users if users is None else users
        sub = self.ds.subset(domain=domain)
        sub = sub[sub["stimulus_id"].astype(str).isin(feats)]

        for uid in sorted(users):
            du = sub[sub["user_id"] == uid]
            stim = du["stimulus_id"].astype(str).unique()
            if len(stim) < n_train + cfg.min_test:
                continue
            rng = user_rng(cfg.split_seed, uid)
            tr_pool, ev_ids = per_user_split(stim, cfg.n_eval, rng)
            agg = self.ds.per_stimulus(du)
            tr = [s for s in tr_pool if s in agg.index][:n_train]
            ev = [s for s in ev_ids if s in agg.index]
            if len(tr) < n_train or len(ev) < cfg.min_test:
                continue
            yield UserUnit(
                fold=fold.index, domain=domain, user_id=int(uid),
                X_train=self.backbone.matrix(feats, tr),
                X_eval=self.backbone.matrix(feats, ev),
                y_train=agg.loc[tr, "overall"].to_numpy(float),
                y_eval=agg.loc[ev, "overall"].to_numpy(float),
                E_train=agg.loc[tr, CORE7].to_numpy(float),
                E_eval=agg.loc[ev, CORE7].to_numpy(float),
                train_ids=tuple(str(s) for s in tr),
                eval_ids=tuple(str(s) for s in ev),
            )

    def group_data(self, users, domain: str, feats, with_dist: bool = False):
        """Image-level (features, population-mean emotions, mean score) for a
        set of users. Used for both the train group and the validation group.

        with_dist also returns the distribution-valued Stage-1 targets, which
        have to be built here because they need the raw per-rater rows.
        """
        d = self.ds.subset(domain=domain, users=users)
        d = d[d["stimulus_id"].astype(str).isin(feats)]
        g = self.ds.per_stimulus(d)
        X = self.backbone.matrix(feats, g.index)
        E = g[CORE7].to_numpy(float)
        y = g["overall"].to_numpy(float)
        if not with_dist:
            return X, E, y
        sd = self.ds.per_stimulus_spread(d).loc[g.index].to_numpy(float)
        hist = self.ds.per_stimulus_hist(d).loc[g.index].to_numpy(float)
        return X, E, y, {"emotion_sd": np.column_stack([E, sd]),
                         "emotion_hist": hist}

    def shared_context(self, fold, domain: str, feats,
                       seed: int = 0, want: list[str] | None = None):
        """Population-level data for this fold/domain, plus fitted mediators.

        Every shared component built here has its hyperparameter selected on
        the validation user group (disjoint from train and test users), never
        on train-group data alone and never on anything a test user touched.
        Stage-1 (mediator) fitting is always ridge, regardless of which
        Stage-2 head is being tested.

        seed  run-level seed for multi-seed averaging; seed=0 keeps the
              original RNG draws exactly.
        """
        want = want or []
        need_dist = any(k in want for k in ("emotion_sd", "emotion_hist", "shuffled35"))

        if need_dist:
            Xg, Eg, yg, Dg = self.group_data(fold.train_users, domain, feats,
                                             with_dist=True)
            Xv, Ev, yv, Dv = self.group_data(fold.val_users, domain, feats,
                                             with_dist=True)
            val_dist = {k: (Xv, v) for k, v in Dv.items()}
        else:
            Xg, Eg, yg = self.group_data(fold.train_users, domain, feats)
            Xv, Ev, yv = self.val_data(fold, domain, feats)
            Dg = val_dist = None

        val_E = (Xv, Ev)

        meds = build_shared_mediators(Xg, Eg, self.cfg, fold.index,
                                      want=want, seed=seed,
                                      val=val_E, yg=yg, val_y=yv, Dg=Dg,
                                      val_dist=val_dist)
        return Xg, Eg, yg, meds

    def val_data(self, fold, domain: str, feats):
        """(X, E, y) of the held-out validation user group for this fold/domain."""
        return self.group_data(fold.val_users, domain, feats)

    def pop_anchor(self, med, Xg, yg, Xv, yv, mediator_name=None):
        """(scaler, w_pop, b_pop) for variant B: the Stage-2 coefficients of
        the pooled training group, in the training group's own units."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from src.modeling.heads import select_alpha_on_val

        Mg, Mv = med.transform(Xg), med.transform(Xv)
        scaler = StandardScaler().fit(Mg)
        Zg, Zv = scaler.transform(Mg), scaler.transform(Mv)
        alphas = np.asarray(self.cfg.ridge_alphas, float)
        a = select_alpha_on_val(Zg, yg, (Zv, yv), alphas, stage="anchor",
                                component="pop_anchor",
                                mediator=mediator_name)
        r = Ridge(alpha=a).fit(Zg, yg)
        return scaler, r.coef_.ravel(), float(r.intercept_)

    def make_personal(self, kind: str, seed: int, variant: str, anchor=None,
                      pop_head=None):
        """Build the personal head for the chosen Stage-2 variant.

        anchor is None for mediators the variant does not apply to (the
        content-free controls), which fall back to a plain head.
        """
        if variant in ("B", "C") and kind in ALPHA_HEADS and anchor is not None:
            scaler, w_pop, b_pop = anchor
            return _PopAnchoredRidge(self.cfg, variant, scaler, w_pop, b_pop,
                                     pop_head, kind=kind)
        if variant == "C" and anchor is not None:
            # C is "fit the head on y - y_pop and add the prediction back",
            # which needs no weight space, so it applies to any head. The
            # wrapped head is built with scale=False because _ResidualHead
            # applies the training-group scaler -- the same one the anchored
            # ridge uses -- so the two rows are standardized identically.
            return _ResidualHead(make_head(kind, self.cfg, seed=seed, scale=False),
                                 pop_head, scaler=anchor[0])
        # B shrinks toward w_pop, which only exists for a linear head.
        return make_head(kind, self.cfg, seed=seed)

    def select_personal_hyperparam(self, fold, domain: str, feats, med, kind: str,
                                   n_train: int, variant: str = "plain",
                                   anchor=None, pop_head=None,
                                   mediator_name=None):
        """Freeze one personal-head hyperparameter for (fold, domain, n_train)
        by mirroring the test protocol inside the validation user group.

        For every validation user: the identical support/eval split
        (`iter_units` with `users=fold.val_users`) is run, a candidate is fit
        on the support set and scored with SROCC on that user's own 50-image
        eval set, and the score is averaged across all validation units. The
        winner is frozen and applied to every test user's personal head at
        this n_train -- no test-group data is touched, and no test user's
        head is chosen by their own held-out performance.

        med   a fitted Mediator (frozen; transforms X_train/X_eval)
        kind  "ridge", "lasso", "elastic" or "mlp"

        A candidate is a penalty for the linear heads and an (lr, epochs)
        pair for the MLP. Both are hashable and both are scored by this one
        loop, so the MLP goes through the same validation protocol as every
        other condition. The reported configuration supplies only one epoch
        budget (500), so its effective search axis is the learning rate.
        """
        cfg = self.cfg
        is_alpha = kind in ALPHA_HEADS
        # a penalty is one number; an MLP candidate is an (lr, epochs) pair.
        # Both are hashable, so one scoring loop carries both.
        cands = ([float(c) for c in head_grid(kind, cfg)] if is_alpha
                 else [(float(lr), int(ep)) for lr, ep in head_grid(kind, cfg)])
        scores = {c: [] for c in cands}

        for unit in self.iter_units(fold, domain, feats, n_train=n_train,
                                    users=fold.val_users):
            M_tr = med.transform(unit.X_train)
            M_ev = med.transform(unit.X_eval)
            for c in cands:
                kw = {"frozen_alpha": c} if is_alpha else {"frozen_lr": c}
                h = self.make_personal(kind, unit.user_id, variant, anchor, pop_head)
                if isinstance(h, (_ResidualHead, _PopAnchoredRidge)):
                    h.fit(M_tr, unit.y_train, X_raw=unit.X_train, **kw)
                    p = h.predict(M_ev, X_raw=unit.X_eval)
                else:
                    p = h.fit(M_tr, unit.y_train, **kw).predict(M_ev)
                scores[c].append(srocc(unit.y_eval, p))

        means = {c: (np.mean(v) if v else -np.inf) for c, v in scores.items()}
        best = max(means.values())
        tied = [c for c, s in means.items() if s >= best - abs(best) * ALPHA_TIE_RTOL]
        # same tie-break direction as select_alpha_on_val for a penalty --
        # the strongest, i.e. the more regularized model. An MLP candidate is
        # conservative in the other direction, so among tied pairs the
        # smallest step and then the shortest budget wins, which is what the
        # lexicographic minimum of an ascending grid gives.
        chosen = max(tied) if is_alpha else min(tied)
        n_units = len(next(iter(scores.values()))) if scores else 0
        note = f"variant={variant} n_train={n_train}"
        if not is_alpha:
            note += f" alpha={cfg.mlp_alpha} (fixed)"
        selection_log.note(
            "stage2_personal", f"personal_{kind}", "val_srocc_mean",
            cands, [means[c] for c in cands], chosen,
            lower_is_better=False,
            axis2="alpha" if is_alpha else "epochs",
            mediator=mediator_name, head=kind,
            n_val_scored=n_units, n_val_kind="user_units",
            n_val_users=len(fold.val_users),
            extra_note=note)
        return chosen

    def _eval_fold_domain(self, fold_index: int, dom: str,
                          mediators: list[str], heads: list[str],
                          n_train: int | None, include_population: bool,
                          include_gt_upper_bound: bool, seed: int,
                          variant: str) -> tuple[list[dict], list[dict]]:
        """Evaluate one (fold, domain) slice.

        Returns (result rows, selection-log rows). The second element is
        every hyperparameter this slice selected, with the grid it was
        selected from -- see src/utils/selection_log.py. It is returned
        rather than written here because this method runs inside a joblib
        worker process when n_jobs > 1, and only the return value comes back.
        """
        cfg = self.cfg
        with selection_log.collecting(
                experiment=getattr(self, "experiment_name", None),
                backbone=getattr(self.backbone, "name", None),
                variant=variant, n_train=n_train or cfg.n_train,
                seed=seed, fold=fold_index, domain=dom) as sel:
            rows = self._eval_fold_domain_inner(
                fold_index, dom, mediators, heads, n_train,
                include_population, include_gt_upper_bound, seed, variant)
        return rows, sel

    def _eval_fold_domain_inner(self, fold_index: int, dom: str,
                                mediators: list[str], heads: list[str],
                                n_train: int | None, include_population: bool,
                                include_gt_upper_bound: bool, seed: int,
                                variant: str) -> list[dict]:
        """The evaluation itself. Split out so the selection buffer above
        wraps every selection, including the ones inside shared_context."""
        cfg = self.cfg
        fold = [f for f in self.split.folds() if f.index == fold_index][0]
        feats = self.backbone.features_for_fold(fold.index)
        want = list(dict.fromkeys(
            list(mediators) + ["identity", "emotion", "pca", "random",
                               "shuffled"]))
        Xg, Eg, yg, meds = self.shared_context(
            fold, dom, feats, seed=seed, want=want)
        n = n_train or cfg.n_train

        print(f"[fold {fold.index}/{dom}] fitting shared components", flush=True)
        # GIAA head. Always fit: it is the population baseline row and
        # also the y_pop that variants A and C are built on.
        # shared component -> hyperparameter from the val group
        Xv, _, yv = self.val_data(fold, dom, feats)
        pop_models = {}
        for h in set(heads) | {"ridge"}:
            base = 100 + fold.index if h == "mlp" else 0
            hseed = base if (h != "mlp" or seed == 0) else base + seed * 1_000_003
            pop_models[h] = make_head(h, cfg, seed=hseed).fit(
                Xg, yg, val=(Xv, yv))
        pop_ridge = pop_models["ridge"]

        # the variant reaches only the mediators listed in the config,
        # so the content-free controls stay content-free
        touched = set(cfg.stage2_variant_mediators)
        if variant == "A":
            for k in list(meds):
                if k in touched:
                    meds[k] = _WithPopFeature(meds[k], pop_ridge)

        # C needs the training-group scaler too, even though it does
        # not use w_pop -- the correction it learns has to live in the
        # same units for every user
        anchors = {}
        if variant in ("B", "C"):
            for mname in mediators:
                if mname in touched:
                    anchors[mname] = self.pop_anchor(
                        meds[mname], Xg, yg, Xv, yv, mediator_name=mname)

        print(f"[fold {fold.index}/{dom}] selecting on "
              f"{len(fold.val_users)} validation users", flush=True)
        # freeze one personal-head hyperparameter per (mediator, head)
        frozen = {}
        for mname in mediators:
            for h in heads:
                frozen[(mname, h)] = self.select_personal_hyperparam(
                    fold, dom, feats, meds[mname], h, n, variant,
                    anchors.get(mname), pop_ridge, mediator_name=mname)

        print(f"[fold {fold.index}/{dom}] evaluating "
              f"{len(fold.test_users)} test users", flush=True)
        rows = []
        # what the frozen hyperparameter produced once it met a real user:
        # epochs run, support size, and how much of the head survived. Kept
        # per (mediator, head) and summarized after the loop, because one row
        # per user would bury the selection log it lives in.
        diag: dict = {}
        for u_idx, unit in enumerate(self.iter_units(fold, dom, feats, n_train=n_train), 1):
            if u_idx % 10 == 0 or u_idx == len(fold.test_users):
                print(f"[fold {fold.index}/{dom}] users "
                      f"{u_idx}/{len(fold.test_users)}", flush=True)
            base = dict(fold=unit.fold, domain=unit.domain,
                        user_id=unit.user_id)

            for h in heads:
                if include_population:
                    p = pop_models[h].predict(unit.X_eval)
                    rows.append({**base, "mediator": "population", "head": h,
                                 "eff_dof": np.nan,
                                 **evaluate(unit.y_eval, p)})

                for mname in mediators:
                    med = meds[mname]
                    M_tr = med.transform(unit.X_train)
                    M_ev = med.transform(unit.X_eval)
                    hseed = (unit.user_id if seed == 0
                            else unit.user_id + seed * 1_000_003)
                    fval = frozen[(mname, h)]
                    head = self.make_personal(h, hseed, variant,
                                              anchors.get(mname), pop_ridge)
                    kw = ({"frozen_alpha": fval} if h in ALPHA_HEADS
                          else {"frozen_lr": fval})
                    if isinstance(head, (_ResidualHead, _PopAnchoredRidge)):
                        head.fit(M_tr, unit.y_train,
                                 X_raw=unit.X_train, **kw)
                        p = head.predict(M_ev, X_raw=unit.X_eval)
                    else:
                        p = head.fit(M_tr, unit.y_train, **kw).predict(M_ev)
                    rows.append({**base, "mediator": mname, "head": h,
                                 "eff_dof": head.effective_dof(),
                                 **evaluate(unit.y_eval, p)})
                    d = diag.setdefault((mname, h), {"n_iter": [], "dof": [],
                                                     "n_support": [], "width": []})
                    d["n_iter"].append(_head_n_iter(head))
                    d["dof"].append(head.effective_dof())
                    d["n_support"].append(len(unit.y_train))
                    d["width"].append(M_tr.shape[1])

            if include_gt_upper_bound:
                # a reference ceiling excluded from every fairness
                # comparison in the paper, not a condition being
                # compared -- keeps its own per-user selection
                head = make_head("ridge", cfg).fit(unit.E_train, unit.y_train)
                p = head.predict(unit.E_eval)
                rows.append({**base, "mediator": "gt_emotion", "head": "ridge",
                             "eff_dof": head.effective_dof(),
                             **evaluate(unit.y_eval, p)})

        for (mname, h), d in diag.items():
            if h in ALPHA_HEADS:
                lr, epochs, alpha = None, None, frozen[(mname, h)]
            else:
                # the frozen MLP value is an (lr, epochs) pair now, so the
                # diagnostic row has to be given both rather than dropping
                # the pair into the lr column
                from src.modeling.heads import split_mlp_candidate
                lr, epochs = split_mlp_candidate(frozen[(mname, h)], cfg)
                alpha = cfg.mlp_alpha
            for name, vals, note_ in (
                    ("n_units", [len(d["n_iter"])], None),
                    ("mediator_width", d["width"], None),
                    ("n_support", d["n_support"], None),
                    ("mean_n_iter", d["n_iter"], f"max_iter={cfg.mlp_max_iter}"),
                    ("frac_hit_max_iter",
                     [float(v >= cfg.mlp_max_iter) for v in d["n_iter"]
                      if v == v], None),
                    ("mean_eff_dof", d["dof"], None)):
                vals = [v for v in vals if v == v]     # drop NaN
                if vals:
                    selection_log.note_fit(
                        "stage2_personal", f"personal_{h}", name,
                        float(np.mean(vals)), mediator=mname, head=h,
                        lr=lr, alpha=alpha, epochs=epochs,
                        extra_note=note_)
        return rows

    def run_grid(self, mediators: list[str], heads: list[str],
                 n_train: int | None = None, include_population: bool = True,
                 include_gt_upper_bound: bool = True,
                 domains: list[str] | None = None, seed: int = 0,
                 stage2_variant: str | None = None,
                 folds: list[int] | None = None) -> pd.DataFrame:
        """Loop (fold, domain, user) x (mediator, head), return per-unit results.

        include_population      add the no-personalization (GIAA) baseline
        include_gt_upper_bound  add the ceiling that uses true emotion ratings
        seed  run-level seed -- every stochastic point (random/shuffled
              mediator) is tied to this seed. seed=0
              reproduces the original single-seed behavior exactly (see
              table1.py, which loops seeds and averages).

        Personal-head hyperparameters are frozen once per (fold, domain,
        mediator, head) by mirroring the test protocol inside the validation
        group (Pipeline.select_personal_hyperparam) and then applied
        identically to every test user -- no test user's head is chosen by
        their own held-out performance, and every mediator goes through the
        same selection code.

        folds  run only these fold indices. Folds share nothing -- separate
        users, separate mediators, separate selection -- so running them as
        separate processes and concatenating the results is identical to
        running them in one, and is how a long grid is spread over cores.

        stage2_variant  plain / A / B / C (defaults to cfg.stage2_variant).
        Every variant is applied to every mediator, Direct included, so a gain
        from the population prior cannot be mistaken for a gain from the
        bottleneck. Run the same grid once per variant and compare.
        """
        cfg = self.cfg
        domains = domains or DOMAINS
        variant = stage2_variant or cfg.stage2_variant
        rows = []
        n_jobs = getattr(cfg, "n_jobs", 1)
        want_folds = None if folds is None else {int(f) for f in folds}
        tasks = [(fold.index, dom) for fold in self.split.folds()
                 if (want_folds is None or fold.index in want_folds)
                 for dom in domains]

        selections: list[dict] = []
        if n_jobs == 1 or len(tasks) <= 1:
            for fold_idx, dom in tasks:
                fold_rows, fold_sel = self._eval_fold_domain(
                    fold_idx, dom, mediators, heads, n_train,
                    include_population, include_gt_upper_bound, seed, variant)
                rows.extend(fold_rows)
                selections.extend(fold_sel)
                if dom == domains[-1]:
                    print(f"  fold {fold_idx} done ({len(rows)} rows)", flush=True)
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_eval_fold_domain_worker)(
                    self, fold_idx, dom, mediators, heads, n_train,
                    include_population, include_gt_upper_bound, seed, variant)
                for fold_idx, dom in tasks
            )
            for fold_rows, fold_sel in results:
                rows.extend(fold_rows)
                selections.extend(fold_sel)
            for fold_idx in sorted(set(f for f, _ in tasks)):
                f_count = sum(len(r) for (f, _), (r, _s) in zip(tasks, results)
                              if f <= fold_idx)
                print(f"  fold {fold_idx} done ({f_count} rows)", flush=True)

        # the caller writes it (see table1.py / efficiency.py); keeping it on
        # the pipeline means every existing caller of run_grid still gets the
        # same DataFrame back and nothing downstream has to change
        self.selection_records = selections
        return pd.DataFrame(rows)

    def collect_user_heads(self, mediator: str = "emotion",
                           domains: list[str] | None = None) -> list[dict]:
        """Fit the head for every unit, keep the fitted models + eval data.

        Feeds the faithfulness experiments (formula swap, weight vs.
        empirical correlation).
        """
        domains = domains or DOMAINS
        store = []
        for fold in self.split.folds():
            feats = self.backbone.features_for_fold(fold.index)
            for dom in domains:
                _, _, _, meds = self.shared_context(fold, dom, feats)
                med = meds[mediator]
                for unit in self.iter_units(fold, dom, feats):
                    M_tr = med.transform(unit.X_train)
                    M_ev = med.transform(unit.X_eval)
                    head = make_head("ridge", self.cfg).fit(M_tr, unit.y_train)
                    store.append(dict(fold=unit.fold, domain=unit.domain,
                                      user_id=unit.user_id, head=head,
                                      M_train=M_tr, M_eval=M_ev, y_eval=unit.y_eval,
                                      E_eval=unit.E_eval))
            print(f"  fold {fold.index} done ({len(store)} units)", flush=True)
        return store


def _head_n_iter(head) -> float:
    """Epochs the personal MLP ran, through whichever variant wrapper is on
    top of it. NaN for a head that does not iterate (ridge, lasso)."""
    inner = getattr(head, "head", head)          # _ResidualHead wraps one
    return float(getattr(inner, "n_iter_", np.nan))


def _eval_fold_domain_worker(pipe: Pipeline, fold_index: int, dom: str,
                             mediators: list[str], heads: list[str],
                             n_train: int | None, include_population: bool,
                             include_gt_upper_bound: bool, seed: int,
                             variant: str) -> tuple[list[dict], list[dict]]:
    res, sel = pipe._eval_fold_domain(
        fold_index, dom, mediators, heads, n_train,
        include_population, include_gt_upper_bound, seed, variant)
    print(f"[fold {fold_index}/{dom}] complete: {len(res)} result rows, "
          f"{len(sel)} selection rows", flush=True)
    return res, sel
