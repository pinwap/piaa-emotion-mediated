"""Head = maps "mediator output" to "this user's beauty score".

The head is the one layer that's personal to a user. Three kinds:

  RidgeHead        linear, interpretable (7 weights = that user's formula)
  SparseLinearHead lasso / elastic net -- linear, and can zero a coefficient
  MLPHead          one hidden layer of cfg.mlp_hidden units, the predictor
                   half of the MLP series

They all train **sequentially**: the mediator is fit and frozen first, then
the head is fit on whatever the mediator outputs. Not end-to-end.

*** where hyperparameters come from ***
Both `fit` methods take an optional `val=(X_val, y_val)`, and that decides
which of the two selection rules applies:

  val given (a SHARED component, e.g. the population/GIAA head) -> the
    hyperparameter is scored on the held-out validation *user group*, which
    is disjoint from both train and test users.
  val=None (a PERSONAL component, i.e. a per-user head) -> selected inside
    that user's own support set only. The validation group is other people,
    so it cannot speak to one user's taste; what matters is that the user's
    evaluation images are never touched, and they aren't.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import effective_dof


class Head(ABC):
    """Common interface for every head."""

    name: str = "head"
    is_linear: bool = False

    @abstractmethod
    def fit(self, M: np.ndarray, y: np.ndarray, val=None) -> "Head":
        """M = mediator output (n_samples, width), y = user's scores.

        val = (M_val, y_val) from the held-out validation user group, for
        shared components only. See module docstring.
        """

    @abstractmethod
    def predict(self, M: np.ndarray) -> np.ndarray:
        ...

    def effective_dof(self) -> float:
        """Effective degrees of freedom -- only defined for linear heads."""
        return np.nan

    def weights(self) -> np.ndarray | None:
        """Coefficients, if linear -- used for interpretability."""
        return None


#: two alphas whose validation MSE differs by less than this (relative) are
#: treated as tied. Floating-point arithmetic is not bit-identical across
#: BLAS builds, so a strict `<` could hand the win to a different alpha on
#: another machine and silently produce a different model. Anything inside
#: this margin is genuinely indistinguishable, so we break the tie by rule
#: instead of by rounding noise. This narrows the cross-platform gap but does
#: not close it: see "What reproducible does and does not promise" in
#: docs/METHODOLOGY.md for what actually holds across machines.
ALPHA_TIE_RTOL = 1e-9


def select_alpha_on_val(X, Y, val, alphas) -> float:
    """Pick the ridge penalty that does best on the held-out validation group.

    Fit on the train-group data, score MSE on the validation group, take the
    winner. Works for multi-output Y (the 7-emotion mediator) as well.

    Ties are broken toward the *strongest* penalty: among alphas that are
    statistically indistinguishable on the validation group, the most
    regularized one is the conservative choice, and picking it by rule makes
    the result reproducible rather than dependent on the last few bits.
    """
    Xv, Yv = val
    alphas = np.asarray(alphas, float)
    mses = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        p = make_pipeline(StandardScaler(), Ridge(alpha=float(a)))
        p.fit(X, Y)
        mses[i] = np.mean((p.predict(Xv) - Yv) ** 2)
    tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
    return float(alphas[tied][-1])          # alphas is ascending -> strongest


class RidgeHead(Head):
    name = "ridge"
    is_linear = True

    def __init__(self, alphas):
        self.alphas = np.asarray(alphas, float)
        self._pipe = None
        self._M_train = None

    def fit(self, M, y, val=None, frozen_alpha=None):
        """frozen_alpha, if given, skips selection entirely -- used when the
        penalty was already chosen once for the whole (fold, domain, n_train)
        by mirroring the test protocol inside the validation group, and is
        then applied identically to every test user (see
        Pipeline.select_personal_hyperparam)."""
        if frozen_alpha is not None:
            self._alpha = float(frozen_alpha)
            self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self._alpha))
            self._pipe.fit(M, y)
        elif val is None:
            self._pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
            self._pipe.fit(M, y)
            self._alpha = float(self._pipe[-1].alpha_)
        else:
            self._alpha = select_alpha_on_val(M, y, val, self.alphas)
            self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self._alpha))
            self._pipe.fit(M, y)
        self._M_train = np.asarray(M, float)
        return self

    def predict(self, M):
        return self._pipe.predict(M)

    @property
    def alpha_(self) -> float:
        return float(self._alpha)

    def effective_dof(self) -> float:
        return effective_dof(self._M_train, self.alpha_)

    def weights(self):
        return self._pipe[-1].coef_.ravel().copy()


class SparseLinearHead(Head):
    """Lasso / ElasticNet personal head -- linear like ridge, but it can set a
    coefficient to exactly zero.

    Worth having as its own row: at small support sizes the ridge head spreads
    a little weight over all 7 emotions, while a sparse head keeps only the
    ones the user's ratings actually support. That is both a different
    bias-variance trade and a more readable formula, which is the property
    this paper is arguing for.

    effective_dof is the number of non-zero coefficients, the standard
    unbiased estimate for the lasso, so it stays comparable with the ridge
    head's trace-based value.
    """

    is_linear = True

    def __init__(self, kind: str, cfg):
        self.name = kind
        self.cfg = cfg
        self.alphas = np.asarray(cfg.sparse_alphas, float)
        self._pipe = None
        self._alpha = np.nan

    def _make(self, alpha: float):
        from sklearn.linear_model import ElasticNet, Lasso
        if self.name == "lasso":
            est = Lasso(alpha=float(alpha), max_iter=self.cfg.sparse_max_iter)
        else:
            est = ElasticNet(alpha=float(alpha),
                             l1_ratio=self.cfg.elastic_l1_ratio,
                             max_iter=self.cfg.sparse_max_iter)
        return make_pipeline(StandardScaler(), est)

    def fit(self, M, y, val=None, frozen_alpha=None):
        if frozen_alpha is not None:
            self._alpha = float(frozen_alpha)
        elif val is None:
            # personal head with no frozen value: pick on the support set
            from sklearn.linear_model import ElasticNetCV, LassoCV
            if self.name == "lasso":
                cv = LassoCV(alphas=self.alphas, max_iter=self.cfg.sparse_max_iter)
            else:
                cv = ElasticNetCV(alphas=self.alphas,
                                  l1_ratio=self.cfg.elastic_l1_ratio,
                                  max_iter=self.cfg.sparse_max_iter)
            p = make_pipeline(StandardScaler(), cv).fit(M, y)
            self._alpha = float(p[-1].alpha_)
        else:
            self._alpha = self._select_on_val(M, y, val)
        self._pipe = self._make(self._alpha)
        self._pipe.fit(M, y)
        return self

    def _select_on_val(self, M, y, val) -> float:
        Mv, yv = val
        mses = np.empty(len(self.alphas))
        for i, a in enumerate(self.alphas):
            p = self._make(a).fit(M, y)
            mses[i] = np.mean((p.predict(Mv) - yv) ** 2)
        tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
        return float(self.alphas[tied][-1])     # ascending -> sparsest

    def predict(self, M):
        return self._pipe.predict(M)

    @property
    def alpha_(self) -> float:
        return float(self._alpha)

    def effective_dof(self) -> float:
        return float(np.count_nonzero(self._pipe[-1].coef_.ravel()))

    def weights(self):
        return self._pipe[-1].coef_.ravel().copy()


def mlp_grid(cfg):
    """Every (learning rate, weight decay) pair the MLP is selected over.

    One grid, used for both stages, so "the MLP's hyperparameters" means the
    same thing in the extractor and in the predictor. Ordered lr-major and
    ascending in both, which is what the tie-break in
    Pipeline.select_personal_hyperparam assumes.
    """
    return tuple((float(lr), float(a))
                 for lr in cfg.mlp_lr_grid for a in cfg.mlp_alpha_grid)


def make_mlp(cfg, lr, alpha, seed: int, scale: bool = True):
    """The one place an MLP is constructed, so the extractor and the predictor
    cannot drift apart.

    early_stopping=False removes the internal 85/15 split, which on a support
    set of 10 ratings would leave one or two validation samples and no usable
    stopping signal. tol=0 with n_iter_no_change=max_iter removes the *other*
    way sklearn stops early -- a training-loss plateau -- so every MLP really
    does train for the same fixed number of epochs.
    """
    from sklearn.neural_network import MLPRegressor

    net = MLPRegressor(
        hidden_layer_sizes=(int(cfg.mlp_hidden),), activation="relu",
        solver="adam", alpha=float(alpha), learning_rate_init=float(lr),
        max_iter=int(cfg.mlp_max_iter), early_stopping=False,
        tol=0.0, n_iter_no_change=int(cfg.mlp_max_iter),
        random_state=int(seed))
    return make_pipeline(StandardScaler(), net) if scale else net


class MLPHead(Head):
    """Personal predictor: mediator output -> one score, one hidden layer.

    Its hyperparameter is the (learning rate, weight decay) pair, and it is
    *not* chosen here. Pipeline.select_personal_hyperparam freezes one pair
    per (fold, domain, mediator, n_train) on the validation user group and
    passes it in as `frozen_hp`, exactly the way the ridge penalty arrives as
    `frozen_alpha`. Choosing it from the user's own support set is the
    behaviour the reviewer asked us to remove, so fit() refuses to run
    without a frozen pair rather than quietly falling back to it.

    `scale=False` is for the anchored case, where the training-group scaler
    has already been applied by _ResidualHead and standardizing a second time
    inside the head would put this row in different units from the ridge row
    beside it.
    """

    name = "mlp"
    is_linear = False

    def __init__(self, cfg, seed: int = 0, scale: bool = True):
        self.cfg, self.seed, self.scale = cfg, int(seed), bool(scale)
        self.model = None
        self._hp = (np.nan, np.nan)

    def fit(self, M, y, val=None, frozen_hp=None, **_):
        if frozen_hp is None:
            if val is None:
                raise ValueError(
                    "MLPHead needs a (learning rate, weight decay) pair frozen "
                    "on the validation group; it must not choose one from the "
                    "user's own support set")
            frozen_hp = self._select_on_val(M, y, val)
        lr, alpha = frozen_hp
        self._hp = (float(lr), float(alpha))
        self.model = make_mlp(self.cfg, lr, alpha, self.seed, self.scale)
        self.model.fit(M, np.asarray(y, float).ravel())
        return self

    def _select_on_val(self, M, y, val):
        """Shared-component path: the GIAA head is fit on the training group
        and scored on the validation group, the same rule select_alpha_on_val
        applies to the ridge GIAA head."""
        Mv, yv = val
        grid = mlp_grid(self.cfg)
        mses = np.array([
            np.mean((make_mlp(self.cfg, lr, a, self.seed, self.scale)
                     .fit(M, np.asarray(y, float).ravel()).predict(Mv)
                     - np.asarray(yv, float).ravel()) ** 2)
            for lr, a in grid])
        return conservative_mlp_hp(
            [g for g, m in zip(grid, mses) if m <= mses.min() * (1.0 + ALPHA_TIE_RTOL)])

    def predict(self, M):
        return np.asarray(self.model.predict(M), float).ravel()

    @property
    def hp_(self) -> tuple:
        return self._hp


def conservative_mlp_hp(tied):
    """Break a tie among (lr, weight decay) pairs the same way
    select_alpha_on_val breaks one among ridge penalties: toward the more
    regularized model. Here that is the smallest step and the strongest decay.
    """
    return min(tied, key=lambda c: (c[0], -c[1]))


#: heads whose hyperparameter is a penalty passed as `frozen_alpha`
#: (everything except the MLP, whose candidate is an (lr, decay) pair and
#: arrives as `frozen_hp`)
ALPHA_HEADS = ("ridge", "lasso", "elastic")


def head_grid(kind: str, cfg):
    """The hyperparameter grid `kind` is selected over."""
    if kind == "ridge":
        return cfg.ridge_alphas
    if kind in ("lasso", "elastic"):
        return cfg.sparse_alphas
    if kind == "mlp":
        return mlp_grid(cfg)
    raise KeyError(f"unknown head '{kind}' (have: ridge, lasso, elastic, mlp)")


def make_head(kind: str, cfg, seed: int = 0, scale: bool = True) -> Head:
    if kind == "ridge":
        return RidgeHead(cfg.ridge_alphas)
    if kind in ("lasso", "elastic"):
        return SparseLinearHead(kind, cfg)
    if kind == "mlp":
        return MLPHead(cfg, seed=seed, scale=scale)
    raise KeyError(f"unknown head '{kind}' (have: ridge, lasso, elastic, mlp)")
