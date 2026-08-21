"""Mediator = the 7-dimensional layer sitting between image features and a
personal score.

Every mediator is fit on **train-user images only**, then frozen and
shared across users -- all the personalization lives in the head.

Mediators used in the paper:

  identity  no mediator (Direct) -- head runs on raw 512-dim features
  emotion   predicts 7 emotions (our proposal) -- a mediator with meaning
  pca       unsupervised 7-dim compression -- controls for "is the gain
            just dimensionality reduction?"
  random    random linear projection to 7 dims -- controls for "does any
            7-dim mediator work?"
  shuffled  emotion predictions shuffled across images -- keeps the
            distribution, destroys the meaning

*** reproducibility note ***
random and shuffled draw from the same per-fold generator, in order: R
first, then the permutation. Reorder or split the generator and the
numbers change even though nothing about the method did.
build_shared_mediators() always draws both, regardless of which mediators
were actually requested.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _shared_ridge(Xg, Yg, alphas, val=None):
    """Ridge for a shared (population-level) mediator.

    The mediator is a shared component, so its penalty is chosen on the
    held-out validation user group when one is available -- those users are
    disjoint from both the train users it is fit on and the test users it is
    scored on. Falls back to RidgeCV's internal generalized CV only if no
    validation data was passed (used by ad-hoc scripts, never by the paper).
    """
    from src.modeling.heads import select_alpha_on_val

    alphas = np.asarray(alphas, float)
    if val is None:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    else:
        m = make_pipeline(StandardScaler(),
                          Ridge(alpha=select_alpha_on_val(Xg, Yg, val, alphas)))
    m.fit(Xg, Yg)
    return m


def _shared_mlp(Xg, Yg, cfg, seed, val):
    """Stage-1 as a one-hidden-layer MLP, hyperparameters from the val group.

    Deliberately the same shape as _shared_ridge: fit on the training group,
    score MSE on the held-out validation users, break ties toward the more
    regularized model, refit the winner. The learning rate and weight decay
    are selected together, because a network with more parameters than
    samples is decided by its penalty, and tuning ridge's penalty over 17
    values while leaving the MLP's fixed would put a handicap in the table
    and call it a model family.
    """
    from src.modeling.heads import ALPHA_TIE_RTOL, conservative_mlp_hp, make_mlp, mlp_grid

    if val is None:
        raise ValueError("a Stage-1 MLP needs the validation group")
    Xv, Yv = val
    grid = mlp_grid(cfg)
    mses = np.array([np.mean((make_mlp(cfg, lr, a, seed).fit(Xg, Yg).predict(Xv) - Yv) ** 2)
                     for lr, a in grid])
    lr, alpha = conservative_mlp_hp(
        [g for g, m in zip(grid, mses) if m <= mses.min() * (1.0 + ALPHA_TIE_RTOL)])
    return make_mlp(cfg, lr, alpha, seed).fit(Xg, Yg)


class _JointNet:
    """features -> h -> 7 concepts -> 1 score, trained under one loss:

        L = MSE(concepts)/2 + w * MSE(score)/2 + L2 penalty

    Koh et al.'s *joint* concept bottleneck, at the population level. The
    score head reads the seven concepts, not the hidden layer, so the score
    gradient is forced through the bottleneck -- that is what makes it joint.
    Hanging the score head off the hidden layer instead would shape the trunk
    while leaving the seven concepts under emotion supervision alone, which
    is sequential training wearing a joint label.

    Joint here means joint across the two *shared* stages, not end-to-end
    into the personal head. It is fit on the training group and frozen, like
    every other Stage-1, and never on a test user's own ratings: a per-user
    Stage-1 would be d*h weights fitted from ~100 samples, and the
    seven-parameters-per-user claim the paper makes would be gone.

    Hand-written because sklearn cannot express a head that reads the
    bottleneck and torch is not a dependency here. Initialization,
    minibatching, the Adam constants and the L2 convention all follow
    sklearn's MLPRegressor, so "500 epochs at this learning rate and weight
    decay" means the same thing in this row as in the sequential row beside
    it. `predict` returns the seven concepts, so EmotionMediator wraps it
    unchanged and Stage-2 cannot tell the two apart.
    """

    B1, B2, EPS, BATCH = 0.9, 0.999, 1e-8, 200

    def __init__(self, d_in, h, k, lr, alpha, w_score, max_iter, seed):
        rng = np.random.RandomState(int(seed))
        self.W1 = self._init(rng, d_in, h)
        self.b1 = np.zeros(h)
        self.W2 = self._init(rng, h, k)
        self.b2 = np.zeros(k)
        self.v = self._init(rng, k, 1).ravel()
        self.c0 = 0.0
        self.lr, self.alpha, self.w = float(lr), float(alpha), float(w_score)
        self.iters, self.rng = int(max_iter), rng
        self.mu = self.sd = None

    @staticmethod
    def _init(rng, fan_in, fan_out):
        """sklearn's Glorot-uniform bound, so both MLP rows start alike."""
        b = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-b, b, (fan_in, fan_out))

    #: penalized parameters follow sklearn: weights yes, intercepts no
    WEIGHTS = ("W1", "W2", "v")
    PARAMS = ("W1", "b1", "W2", "b2", "v", "c0")

    def _forward(self, Z):
        H = np.maximum(Z @ self.W1 + self.b1, 0.0)
        C = H @ self.W2 + self.b2
        return H, C, C @ self.v + self.c0

    def _grads(self, Z, Cp, yp):
        n, k = Cp.shape
        H, C, yh = self._forward(Z)
        dC = (C - Cp) / (n * k)                     # d/dC of MSE(concepts)/2
        dy = self.w * (yh - yp) / n                 # d/dyh of w*MSE(score)/2
        dC = dC + np.outer(dy, self.v)              # score routed through the bottleneck
        g = {"v": C.T @ dy, "c0": float(dy.sum()),
             "W2": H.T @ dC, "b2": dC.sum(0)}
        dH = (dC @ self.W2.T) * (H > 0)
        g["W1"] = Z.T @ dH
        g["b1"] = dH.sum(0)
        for w in self.WEIGHTS:                      # L2, sklearn's scaling
            g[w] = g[w] + self.alpha * getattr(self, w) / n
        return g

    def fit(self, X, Cp, yp):
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        Cp = np.asarray(Cp, float)
        yp = np.asarray(yp, float).ravel()
        n = len(Z)
        bs = min(self.BATCH, n)

        m = {k: np.zeros_like(np.atleast_1d(getattr(self, k)), float) for k in self.PARAMS}
        v = {k: np.zeros_like(np.atleast_1d(getattr(self, k)), float) for k in self.PARAMS}
        t = 0
        for _ in range(self.iters):
            order = self.rng.permutation(n)
            for s in range(0, n, bs):
                idx = order[s:s + bs]
                g = self._grads(Z[idx], Cp[idx], yp[idx])
                t += 1
                for k in self.PARAMS:
                    m[k] = self.B1 * m[k] + (1 - self.B1) * g[k]
                    v[k] = self.B2 * v[k] + (1 - self.B2) * np.square(g[k])
                    step = (self.lr * (m[k] / (1 - self.B1 ** t))
                            / (np.sqrt(v[k] / (1 - self.B2 ** t)) + self.EPS))
                    cur = getattr(self, k)
                    # c0 is a plain float; its moment buffers are 1-element
                    # arrays, so unwrap the step before subtracting
                    setattr(self, k, cur - (float(np.ravel(step)[0])
                                            if np.isscalar(cur) else step))
        return self

    def loss(self, X, Cp, yp):
        """The objective itself, for selecting on the validation group."""
        Cp = np.asarray(Cp, float)
        yp = np.asarray(yp, float).ravel()
        _, C, yh = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return 0.5 * np.mean((C - Cp) ** 2) + 0.5 * self.w * np.mean((yh - yp) ** 2)

    def predict(self, X):
        _, C, _ = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return C


def _shared_joint(Xg, Eg, yg, cfg, seed, val, yv):
    """Joint Stage-1, hyperparameters selected on the validation user group.

    Same protocol as the other two Stage-1s -- fit on the training group,
    score on held-out users, tie-break toward the more regularized model --
    scored on this model's own objective, which is the combined one it is
    trained on. (The ridge and sequential-MLP extractors are scored on
    emotion MSE for the same reason: it is what they are fitted to.)
    """
    from src.modeling.heads import ALPHA_TIE_RTOL, conservative_mlp_hp, mlp_grid

    if val is None or yv is None:
        raise ValueError("joint Stage-1 needs the validation group's "
                         "features, emotions and mean scores")
    Xv, Ev = val
    grid = mlp_grid(cfg)

    def build(lr, alpha):
        return _JointNet(Xg.shape[1], int(cfg.mlp_hidden), Eg.shape[1],
                         lr=lr, alpha=alpha, w_score=cfg.joint_score_weight,
                         max_iter=cfg.mlp_max_iter, seed=seed)

    losses = np.array([build(lr, a).fit(Xg, Eg, yg).loss(Xv, Ev, yv)
                       for lr, a in grid])
    lr, alpha = conservative_mlp_hp(
        [g for g, l in zip(grid, losses) if l <= losses.min() * (1.0 + ALPHA_TIE_RTOL)])
    return build(lr, alpha).fit(Xg, Eg, yg)


class Mediator(ABC):
    name: str = "mediator"
    label: str = "Mediator"

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map image features to the mediator's output."""


class IdentityMediator(Mediator):
    """Direct -- no mediator, raw features go straight to the head."""
    name, label = "identity", "Direct"

    def transform(self, X):
        return np.asarray(X, float)


class EmotionMediator(Mediator):
    """Predicts 7 emotions from an image, using a model shared across users."""
    name, label = "emotion", "Hybrid (ours)"

    def __init__(self, model):
        self.model = model

    def transform(self, X):
        return self.model.predict(X)


class PCAMediator(Mediator):
    name, label = "pca", "PCA"

    def __init__(self, pca: PCA):
        self.pca = pca

    def transform(self, X):
        return self.pca.transform(X)


class RandomMediator(Mediator):
    name, label = "random", "Random"

    def __init__(self, R: np.ndarray):
        self.R = R

    def transform(self, X):
        return np.asarray(X, float) @ self.R


class ShuffledMediator(Mediator):
    """Emotion predictions shuffled across images -- realistic values, wrong image."""
    name, label = "shuffled", "Shuffled"

    def __init__(self, model):
        self.model = model

    def transform(self, X):
        return self.model.predict(X)


def build_shared_mediators(Xg: np.ndarray, Eg: np.ndarray, cfg, fold_index: int,
                           want: list[str] | None = None,
                           seed: int = 0,
                           val: tuple | None = None,
                           yg: np.ndarray | None = None,
                           val_y: np.ndarray | None = None,
                           Dg: dict | None = None,
                           val_dist: dict | None = None) -> dict[str, Mediator]:
    """Build every mediator from train-user data (population-level images).

    Stage-1 is ridge by default, independent of which Stage-2 head is tested
    against it, so the ridge and MLP rows would otherwise differ only in the
    head. Two mediators make Stage-1 its own axis instead: "emotion_mlp"
    (one hidden layer, still fitted to the emotions alone) and
    "emotion_joint" (one hidden layer, fitted to emotions and score together
    with the score read off the bottleneck). Paired with the MLP Stage-2
    head these are the MLP -> MLP sequential and MLP -> MLP joint rows.

    Xg  features of images train users rated (n_img, d)
    Eg  population-mean emotion ratings for those images (n_img, 7)
    seed  run-level seed for multi-seed averaging (random/shuffled are
          stochastic); seed=0 reproduces the original RNG exactly.
    val  (X_val, E_val) from the held-out validation user group; the ridge
         penalty of every fitted mediator is selected on it. The emotion and
         shuffled mediators get the exact same treatment, so the control
         differs from the real thing only in the labels it saw.

    Random-draw order is fixed for reproducibility, see module docstring.
    """
    K = cfg.mediator_width
    want = want or ["identity", "emotion", "pca", "random", "shuffled"]

    # fixed RNG order: R first, then the permutation -- don't reorder, or the
    # published random and shuffled numbers move even though nothing about
    # the method did.
    rng_seed = fold_index if seed == 0 else fold_index + seed * 1_000_003
    rng = np.random.default_rng(rng_seed)
    R = rng.standard_normal((Xg.shape[1], K)) / np.sqrt(Xg.shape[1])
    perm = rng.permutation(len(Eg))

    # 35-wide controls for emotion_hist: same construction as the 7-wide
    # ones, drawn unconditionally right after them so the draw order stays
    # fixed regardless of `want`, and from the same rng so they are
    # reproducible the same way. Width 35 to match emotion_hist, not
    # emotion, since these exist to ask "is emotion_hist's edge about its
    # 35 numbers, or about what they encode?"
    R35 = rng.standard_normal((Xg.shape[1], 35)) / np.sqrt(Xg.shape[1])
    perm35 = rng.permutation(len(Eg))

    out: dict[str, Mediator] = {}
    if "identity" in want:
        out["identity"] = IdentityMediator()
    if "emotion" in want:
        out["emotion"] = EmotionMediator(
            _shared_ridge(Xg, Eg, cfg.ridge_alphas, val))
    if "pca" in want:
        out["pca"] = PCAMediator(PCA(n_components=K, random_state=0).fit(Xg))
    if "random" in want:
        out["random"] = RandomMediator(R)
    if "shuffled" in want:
        out["shuffled"] = ShuffledMediator(
            _shared_ridge(Xg, Eg[perm], cfg.ridge_alphas, val))

    # --- Stage-1 as its own axis: the MLP series ------------------------
    # Same seven concepts, same targets, same validation-group selection
    # protocol -- only the model family that produces them changes. These
    # draw from their own RNG, seeded off rng_seed, so adding or removing
    # them cannot move the random/shuffled numbers above.
    if "emotion_mlp" in want:
        out["emotion_mlp"] = EmotionMediator(
            _shared_mlp(Xg, Eg, cfg, rng_seed, val))
    if "emotion_joint" in want:
        if yg is None or val_y is None:
            raise ValueError("emotion_joint needs the training- and "
                             "validation-group mean scores (yg, val_y)")
        out["emotion_joint"] = EmotionMediator(
            _shared_joint(Xg, Eg, yg, cfg, rng_seed, val, val_y))

    # --- distribution-valued Stage-1 ------------------------------------
    # Same seven named concepts, but Stage-1 predicts how the raters were
    # spread over the scale instead of only where they landed on average.
    # The bottleneck is still "the 7 emotions", so Stage-2 stays readable:
    #   emotion_sd    7 means + 7 standard deviations          (14 wide)
    #   emotion_hist  7 emotions x 5 rating bins               (35 wide)
    # Dg is supplied by the caller because it has to be built from the raw
    # per-rater rows, which this module never sees.
    for key in ("emotion_sd", "emotion_hist"):
        if key in want:
            if Dg is None or key not in Dg:
                raise ValueError(f"{key} needs Dg['{key}'] (per-image targets)")
            out[key] = EmotionMediator(
                _shared_ridge(Xg, Dg[key], cfg.ridge_alphas,
                              (val_dist or {}).get(key)))

    if "pca35" in want:
        out["pca35"] = PCAMediator(PCA(n_components=35, random_state=0).fit(Xg))
    if "random35" in want:
        out["random35"] = RandomMediator(R35)
    if "shuffled35" in want:
        if Dg is None or "emotion_hist" not in Dg:
            raise ValueError("shuffled35 needs Dg['emotion_hist']")
        out["shuffled35"] = ShuffledMediator(
            _shared_ridge(Xg, Dg["emotion_hist"].values[perm35] if hasattr(Dg["emotion_hist"], "values")
                         else Dg["emotion_hist"][perm35],
                         cfg.ridge_alphas, (val_dist or {}).get("emotion_hist")))

    return out
