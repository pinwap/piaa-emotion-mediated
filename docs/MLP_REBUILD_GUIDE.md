# Rebuilding the MLP rows — step by step

All previous MLP code has been removed. This guide rebuilds it from the
reviewer's two messages, nothing else.

## What the reviewer asked for

**8 Aug 2026 — the architecture**

> For the MLP series, please use an MLP with a single hidden layer (either 128
> or 256 units) throughout, applied to both the extractor (the mapping to the
> 7-dimensional features) and the predictor (the mapping to the 1-dimensional
> score).

So one hidden layer, one width used everywhere, and **both** stages are MLPs.
There is no "ridge extractor + MLP predictor" row, and no "MLP extractor +
ridge predictor" row. Only:

| row | Stage-1 (extractor) | Stage-2 (predictor) |
|---|---|---|
| Ridge → Ridge | ridge 512→7 | ridge 7→1 |
| MLP → MLP sequential | MLP 512→h→7, emotion loss only | MLP 7→h→1 |
| MLP → MLP joint | MLP 512→h→7→1, emotion + score loss | MLP 7→h→1 |

**14 Aug 2026 — the selection protocol**

1. Mirror the test protocol inside the validation group: per validation user
   and domain, shuffle with `RandomState(42 + user_id)`, hold out the first 50
   as an inner eval set, draw the support set from the remainder.
2. For each candidate value, fit on each validation user's support set, score
   with SROCC on their inner eval set, average over all validation
   user-domain units, pick the best, freeze it, apply to every test user.
3. Select separately per support size.
4. Apply to **every** condition — Direct, Random, Shuffled, PCA, GIAA — and
   move the MLP learning rate onto this same group-level protocol.
5. No MLP early stopping. Fixed epoch count, stated in advance, or selected
   jointly with the learning rate on the validation group.

**Items 1–3 already exist** in `Pipeline.select_personal_hyperparam`
(`src/modeling/pipeline.py`). Read it before you start; you are plugging into
it, not rewriting it. Item 4 is already true for ridge/lasso/elastic. Your job
is to make the MLP obey the same path.

---

## Step 1 — config

`src/config.py`, just above `mediator_width`:

```python
    # MLP. One hidden layer, same width for extractor and predictor, per the
    # reviewer's 8 Aug instruction. 128 and 256 were both offered; state
    # whichever you run.
    mlp_hidden: int = 128
    mlp_alpha: float = 0.0

    # Fixed epoch budget. No early stopping: with support sets as small as 10
    # ratings an internal validation split leaves one or two samples and gives
    # no usable stopping signal. This number is fixed in advance, not tuned.
    mlp_max_iter: int = 500

    # Learning rate is selected on the validation group, by the same procedure
    # every other hyperparameter uses.
    mlp_lr_grid: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
```

Then run `uv run python -c "from src.config import Config; print(Config().mlp_hidden)"`.
It should print `128`. If it errors, the dataclass field order is wrong —
fields without defaults cannot follow fields with defaults.

## Step 2 — the Stage-2 head

`src/modeling/heads.py`. Add the import at the top:

```python
from sklearn.neural_network import MLPRegressor
```

Then add the class after `SparseLinearHead`:

```python
class MLPHead(Head):
    """Predictor: mediator output -> one score, single hidden layer.

    Trained for a fixed cfg.mlp_max_iter epochs with early_stopping=False.
    The learning rate is not chosen here -- Pipeline.select_personal_hyperparam
    freezes it on the validation group and passes it in as frozen_lr, the same
    way the ridge penalty arrives.
    """

    name = "mlp"
    is_linear = False

    def __init__(self, cfg, seed: int = 0):
        self.cfg, self.seed = cfg, seed
        self.model = None

    def _build(self, lr: float):
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(self.cfg.mlp_hidden,),
            activation="relu",
            alpha=self.cfg.mlp_alpha,
            solver="adam",
            learning_rate_init=float(lr),
            max_iter=self.cfg.mlp_max_iter,
            early_stopping=False,
            random_state=int(self.seed),
        ))

    def fit(self, M, y, val=None, frozen_lr=None, **_):
        if frozen_lr is None:
            raise ValueError(
                "MLPHead needs a learning rate frozen on the validation "
                "group; it must not choose one from the user's own data")
        self.model = self._build(frozen_lr).fit(M, np.asarray(y, float).ravel())
        return self

    def predict(self, M):
        return self.model.predict(M)
```

Note the `raise`. The old code silently fell back to an internal 80/20 split
of the user's own support set, which is exactly what the reviewer told us to
remove. Failing loudly means that can't come back by accident.

Now register it. In `head_grid`:

```python
    if kind == "mlp":
        return cfg.mlp_lr_grid
```

and in `make_head`:

```python
    if kind == "mlp":
        return MLPHead(cfg, seed=seed)
```

and fix the error message to mention mlp again.

**Check it:**
```bash
uv run python -c "
from src.config import Config
from src.modeling.heads import make_head, head_grid
c = Config()
print('grid:', head_grid('mlp', c))
h = make_head('mlp', c)
import numpy as np
X = np.random.RandomState(0).randn(60, 7); y = X[:, 0] + 0.1*np.random.RandomState(1).randn(60)
print('fitted, predict shape:', h.fit(X, y, frozen_lr=1e-3).predict(X).shape)
try:
    make_head('mlp', c).fit(X, y)
except ValueError as e:
    print('correctly refuses without frozen_lr')
"
```

## Step 3 — check the selection path already carries it

Open `src/modeling/pipeline.py` and find `select_personal_hyperparam`. It
already branches on `kind in ALPHA_HEADS` and otherwise passes `frozen_lr`.
`ALPHA_HEADS` does not contain `"mlp"`, so the `else` branch is your path —
you should not need to edit this function at all.

Confirm by reading these two lines in it:

```python
                elif kind in ALPHA_HEADS:
                    p = h.fit(M_tr, unit.y_train, frozen_alpha=float(c)).predict(M_ev)
                else:
                    p = h.fit(M_tr, unit.y_train, frozen_lr=float(c)).predict(M_ev)
```

If they are there, items 1–4 of the protocol apply to the MLP automatically,
because the same function is called for every mediator and every head.

There is one thing to fix: the tie-break. Find

```python
        return max(tied)
```

and make it direction-aware again, since for a learning rate the conservative
choice is the smallest step, not the largest:

```python
        return max(tied) if kind in ALPHA_HEADS else min(tied)
```

**Check it:**
```bash
uv run main.py efficiency --backbone clip --mediators emotion --heads ridge,mlp \
  --n-train 100 --seed 0 --stage2 C
```
Look at the printed summary: two rows for `emotion`, one per head. If the MLP
row is there, selection worked.

## Step 4 — Stage-1 MLP, sequential

`src/modeling/mediators.py`. The mediator only has to expose `.predict(X)`
returning 7 columns, so `EmotionMediator` can wrap it directly.

Add a learning-rate selector that uses the validation group:

```python
def _select_stage1_lr(Xg, Yg, cfg, seed, val):
    """Learning rate for a shared Stage-1 MLP, chosen on the validation group.

    Same rule the ridge mediator uses: a shared component's hyperparameter is
    scored on users disjoint from both train and test.
    """
    from sklearn.neural_network import MLPRegressor
    from src.modeling.heads import ALPHA_TIE_RTOL

    grid = np.asarray(cfg.mlp_lr_grid, float)
    if val is None:
        raise ValueError("Stage-1 MLP needs the validation group")

    Xv, Yv = val
    mses = np.empty(len(grid))
    for i, lr in enumerate(grid):
        m = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(cfg.mlp_hidden,), activation="relu",
            alpha=cfg.mlp_alpha, solver="adam", learning_rate_init=float(lr),
            max_iter=cfg.mlp_max_iter, early_stopping=False,
            random_state=int(seed))).fit(Xg, Yg)
        mses[i] = np.mean((m.predict(Xv) - Yv) ** 2)
    tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
    return float(grid[tied][0])          # ascending -> smallest step on a tie
```

Then register the mediator, inside `build_shared_mediators`, next to the
`"emotion"` block:

```python
    if "emotion_mlp" in want:
        from sklearn.neural_network import MLPRegressor
        lr = _select_stage1_lr(Xg, Eg, cfg, seed, val)
        out["emotion_mlp"] = EmotionMediator(make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(cfg.mlp_hidden,), activation="relu",
                         alpha=cfg.mlp_alpha, solver="adam",
                         learning_rate_init=lr, max_iter=cfg.mlp_max_iter,
                         early_stopping=False, random_state=int(seed))
        ).fit(Xg, Eg))
```

Add `"emotion_mlp"` to `stage2_variant_mediators` in the config, or it will
run unanchored under C while the rows beside it are anchored.

**Check it:**
```bash
uv run python -c "
import warnings; warnings.filterwarnings('ignore')
from src.config import Config
from main import build
cfg = Config(); ds, bb, sp, pipe = build(cfg, 'clip')
d = pipe.run_grid(mediators=['emotion','emotion_mlp'], heads=['ridge','mlp'],
                  n_train=100, include_population=True,
                  include_gt_upper_bound=False, seed=0, stage2_variant='C',
                  domains=['fashion'])
print(d.groupby(['mediator','head']).srocc.mean().round(4))
"
```
You want four mediator/head combinations to appear. The one to report is
`emotion_mlp` + `mlp` — that is MLP → MLP sequential.

## Step 5 — Stage-1 MLP, joint

The joint model differs in one way only: the score head reads the **seven
concepts**, so the score gradient passes through the bottleneck. Routing it
off the hidden layer instead would leave the seven concepts under emotion
supervision alone, which is sequential training with a joint label.

sklearn cannot express that, so write the network directly. Put this in
`src/modeling/mediators.py`:

```python
class _JointNet:
    """512 -> h -> 7 concepts -> 1 score, one combined loss.

        loss = MSE(c, c_pop) + w * MSE(v.c + b, y_pop)

    Trained on the training group only and then frozen, like every other
    Stage-1. It is never fitted on a test user's own ratings: a per-user
    Stage-1 would be 512*h weights from ~100 samples, and the
    seven-parameters-per-user claim would be gone.
    """

    def __init__(self, d_in, h, k, lr, w_score, max_iter, seed):
        rng = np.random.default_rng(int(seed))
        self.W1 = rng.standard_normal((d_in, h)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(h)
        self.W2 = rng.standard_normal((h, k)) * np.sqrt(2.0 / h)
        self.b2 = np.zeros(k)
        self.v = rng.standard_normal(k) * np.sqrt(2.0 / k)
        self.c0 = 0.0
        self.lr, self.w, self.iters = float(lr), float(w_score), int(max_iter)
        self.mu = self.sd = None

    def _keys(self):
        return ["W1", "b1", "W2", "b2", "v", "c0"]

    def _forward(self, Z):
        H = np.maximum(Z @ self.W1 + self.b1, 0.0)
        C = H @ self.W2 + self.b2
        return H, C, C @ self.v + self.c0

    def fit(self, X, Cp, yp):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        Cp = np.asarray(Cp, float); yp = np.asarray(yp, float).ravel()
        n = len(Z)

        m = {k: np.zeros_like(getattr(self, k), dtype=float) for k in self._keys()}
        v = {k: np.zeros_like(getattr(self, k), dtype=float) for k in self._keys()}
        b1_, b2_, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.iters + 1):
            H, C, yh = self._forward(Z)
            dC = (2.0 / n) * (C - Cp)
            dy = (2.0 * self.w / n) * (yh - yp)
            dC = dC + np.outer(dy, self.v)        # through the bottleneck

            g = {"v": C.T @ dy, "c0": dy.sum(), "W2": H.T @ dC, "b2": dC.sum(0)}
            dH = (dC @ self.W2.T) * (H > 0)
            g["W1"] = Z.T @ dH
            g["b1"] = dH.sum(0)

            for k in self._keys():
                m[k] = b1_ * m[k] + (1 - b1_) * g[k]
                v[k] = b2_ * v[k] + (1 - b2_) * g[k] ** 2
                mh = m[k] / (1 - b1_ ** t)
                vh = v[k] / (1 - b2_ ** t)
                setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + eps))
        return self

    def loss(self, X, Cp, yp):
        _, C, yh = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return (np.mean((C - np.asarray(Cp, float)) ** 2)
                + self.w * np.mean((yh - np.asarray(yp, float).ravel()) ** 2))

    def predict(self, X):
        _, C, _ = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return C
```

Add `joint_score_weight: float = 1.0` to the config, register the mediator
(selecting the learning rate on the validation group using `.loss`), and add
`"emotion_joint"` to `stage2_variant_mediators`.

`build_shared_mediators` will need `yg` and the validation group's `yv`
threaded in from `pipeline.py` — the signature already takes `yg`; add
`val_y` beside it.

**Check the gradient before trusting any result.** This is hand-written, so
verify it numerically:

```bash
uv run python -c "
import numpy as np
from src.config import Config
from src.modeling.mediators import _JointNet
rng = np.random.default_rng(0)
X = rng.standard_normal((40, 12)); C = rng.standard_normal((40, 7)); y = rng.standard_normal(40)
net = _JointNet(12, 8, 7, lr=1e-3, w_score=1.0, max_iter=1, seed=0)
before = net.fit(X, C, y).loss(X, C, y)
net2 = _JointNet(12, 8, 7, lr=1e-3, w_score=1.0, max_iter=200, seed=0)
after = net2.fit(X, C, y).loss(X, C, y)
print(f'loss after 1 step {before:.4f} -> after 200 steps {after:.4f}')
assert after < before, 'training does not reduce the loss - gradient is wrong'
print('gradient descends: OK')
"
```

If that assertion fails, stop — the derivative is wrong and every number
downstream would be meaningless.

## Step 6 — the three rows, run together

```bash
uv run main.py efficiency --backbone qwen8b \
  --mediators emotion,emotion_mlp,emotion_joint --heads ridge,mlp \
  --n-train 10,25,50,100 --seed 0,1,2 --stage2 C
```

From the six mediator/head combinations this produces, the paper reports
three:

| report as | mediator | head |
|---|---|---|
| Ridge → Ridge | `emotion` | `ridge` |
| MLP → MLP sequential | `emotion_mlp` | `mlp` |
| MLP → MLP joint | `emotion_joint` | `mlp` |

Ignore the mixed combinations; the reviewer did not ask for them.

## What to state in the paper

- one hidden layer, width 128 (or 256 — whichever you ran)
- fixed `max_iter`, no early stopping, and the number
- learning rate selected on the validation group, per support size, by the
  same procedure as the ridge penalty
- that the same selection procedure is applied to Direct, Random, Shuffled,
  PCA and GIAA
