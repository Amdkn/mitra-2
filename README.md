---
license: apache-2.0
tags:
- tabular
- tabular-classification
- tabular-regression
- foundation-model
- arxiv:2609.04540
---

# mitra-finetune

Fine-tuning for the second-generation pretrained [Mitra v2](https://huggingface.co/autogluon/mitra-classifier-2) tabular foundation-model checkpoints.
The [Mitra-v2 Technical Report](https://arxiv.org/abs/2609.04540) (also on the [Hub](https://huggingface.co/autogluon/mitra-finetune/blob/main/Mitra_v2_Technical_Report.pdf)) describes the model and the evaluation behind the numbers below.

Point at a checkpoint and fit:

```python
from mitra_finetune import MitraFinetune

model = MitraFinetune(checkpoint_dir="checkpoints/")
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)
```

## The recipe

One `MitraFinetune` fit is one Mitra fine-tuning run: a standard 50-step
full fine-tune (lr 1e-5, warmup 10, weight decay 0.3) executed as an AutoGluon
8-fold bagged fit, capped at `time_limit` seconds (default 3,600). The recipe is
frozen: it was selected on one Mitra checkpoint and prospectively confirmed on a
later checkpoint of the same pretraining run and on a held-out evaluation
fold. Measured wall clock per evaluation unit (bagged fine-tune plus
prediction) on the released 38-dataset TabArena classification artifacts
(H100): 0.3 h mean per dataset, 1.4 h max (APSFailure).

Wide tables (more than 256 features) are automatically narrowed before
fitting: top-K feature selection for classification (dtype-gated) and a
truncated-SVD projection to 256 components for regression (see
`feature_selection.py`; override the budget with `MITRA_CLS_MAX_FEATURES` /
`MITRA_REG_MAX_FEATURES` and the method with `MITRA_FS_METHOD=select|svd`),
and classification beyond the checkpoint's native 10-class head
runs through a hierarchical label decomposition (`hierarchy.py`).

For large tables the in-context support is both capped and, on binary
classification, class-balanced. Fine-tuning uses a 16,384-row support cap for
classification and up to 20,480 rows for regression; prediction draws its
in-context support up to 16,384 rows on binary classification, 32,768 on
multiclass classification and 32,768 on regression. On binary tasks the
prediction-time support subsample is class-balanced rather than uniformly
random. The binary prediction cap is deliberately conservative: it keeps every
bag fold fast enough that AutoGluon's time-limit projection never truncates
the bag on large tasks, which scored better end to end than wider contexts.
These are frozen defaults; `MITRA_SUPPORT_CAP`, `MITRA_PREDICT_SUPPORT_CAP`,
and `MITRA_SUPPORT_SELECT` override them.

Two further task-conditioned rules are part of the frozen configuration and
apply uniformly (no per-dataset selection): on binary tasks whose training
table has at most 16,384 rows the fine-tuning learning rate is 3e-6 instead of
1e-5, and after the bagged fit each bag child predicts the test rows with its
full outer training table as in-context support, that is its fit fold plus its
own held-out fold ("heldout in support"). The held-out labels are used only as
fine-tuning validation and as support rows at prediction time; no test
information is involved and no extra training is done. The rule also applies
when an external validation set is passed to `fit`: each child still validates
on the external set, and its own held-out fold is added to its support at
prediction time. Set `MITRA_HELDOUT_IN_SUPPORT=0` to disable it. A separate,
off-by-default switch, `MITRA_VAL_IN_SUPPORT=1`, additionally adds the external
validation rows to every child's prediction-time support (the train+val context
policy of the TALENT boards); validation predictions themselves are always
computed with train-only support.

The reported TabArena numbers (overall Elo 1774.6, classification 1756.3,
regression 1985.6 under the TabArena 1h protocol with default configurations)
were produced with exactly these defaults plus the TabArena 1h protocol, which
is a benchmark setting rather than part of the recipe: a 3,600 s task time
limit, a 250 s fine-tuning budget per bag child, and keeping the already fitted
children as the bag when the task limit hits. The two protocol controls are
implemented in this package (`patches.py`, applied at fit time to stock
AutoGluon) and switched on with environment variables; they are off unless set:

```bash
export MITRA_FT_BUDGET_S=250   # fine-tuning budget per bag child, seconds
export MITRA_BAG_SALVAGE=1     # keep fitted children when the task time limit hits
```

## Installation

Requires Python 3.11 to 3.13, a CUDA GPU, AutoGluon ≥ 1.6 with the Mitra
extra, and the `tabarena` package: the fit runs through TabArena's bagged
AutoGluon wrapper, the exact protocol behind the reported numbers.

```bash
# 1. AutoGluon with the Mitra extra, plus TabArena's execution wrapper
pip install "autogluon.tabular[mitra]>=1.6" "tabarena>=0.1.0"
# 2. Flash-attention (optional but required for realistic speed; prebuilt wheel strongly recommended)
pip install flash-attn --no-build-isolation
# 3. This package. The Hub's git server does not support pip's partial clone
#    (`pip install git+https://...` fails), so clone first, or use uv:
#    `uv pip install git+https://huggingface.co/autogluon/mitra-finetune`
git clone https://huggingface.co/autogluon/mitra-finetune
pip install ./mitra-finetune
```

Note on torch: `autogluon.tabular[mitra]` may resolve to the newest
torch; if you rely on a prebuilt flash-attn wheel, pin torch to the
version your wheel was built against (we use `torch==2.9.1+cu128`) after
installing AutoGluon.

This package fine-tunes the second-generation Mitra (v2) checkpoints, hosted
on the Hugging Face Hub: [autogluon/mitra-classifier-2](https://huggingface.co/autogluon/mitra-classifier-2)
for classification and [autogluon/mitra-regressor-2](https://huggingface.co/autogluon/mitra-regressor-2)
for regression.
Download one (for example
`hf download autogluon/mitra-classifier-2 --local-dir ckpt/`)
and point `checkpoint_dir` at it. You can pass either a raw `.pt` state dict
(converted automatically once to a cached `<ckpt>.pt.ag16/` directory in
`Tab2D.save_pretrained` format) or such a directory directly. AutoGluon 1.6
removed the `state_dict_*` hyperparameters; custom weights load via
`hf_model=<local dir>`, which this package handles for you.

Flash-attn is optional. The fine-tuning loop runs on torch's
`scaled_dot_product_attention` either way (as fast per step as flash-attn 2 on
an H100, 1.3 to 1.7 times faster on an RTX PRO 6000 Blackwell); prediction
uses flash-attn when it is installed, which is faster and lighter on memory at
prediction shapes. A notice is logged at fit time if it is missing.

For development, install the clone in editable mode:

```bash
git clone https://huggingface.co/autogluon/mitra-finetune
cd mitra-finetune
pip install -e .
```

## API

```python
MitraFinetune(
    checkpoint_dir,            # HF weights dir (config.json + model.safetensors), a .pt file, or a directory with one .pt
    problem_type="classification",  # or "regression"
    time_limit=3600,           # fit budget, seconds (shared by the 8 bag children)
    eval_metric=None,          # AutoGluon metric for validation checkpointing; defaults to log_loss (classification) / RMSE (regression); the reported binary tasks used "roc_auc"
    device="cuda",
    random_state=0,
    num_bag_folds=8,
)
```

- `fit(X, y, X_val=None, y_val=None)`: stores the data. Without an
  external validation set, validation comes from the bagged fit itself
  (out-of-fold predictions).
- `predict_proba(X_test)`: runs the fine-tune as an AutoGluon 8-fold
  bagged fit in its own subprocess and returns test probabilities.
  (Mitra is an in-context learner, so the fit executes at prediction
  time; the out-of-fold validation probabilities are exposed as
  `model.val_proba_`.)
- `predict(X_test)`: argmax of `predict_proba` (classification), or
  continuous point predictions (regression): the mean of the predicted
  distribution over the checkpoint's 1,000 target bins.
- `predict_distribution(X_test)` (regression): the same bagged fit as
  `predict`, returning the predicted distribution itself as a
  `RegressionDistribution` (below); `predict(X_test, output_type="full")` is
  an alias. Call it instead of `predict` when you need both and read the
  point predictions from `dist.point_prediction`.

## Distributional regression

The regressor casts regression as classification over 1,000 target bins, so
each bag child predicts a histogram per row and `predict` reports its mean.
`predict_distribution` returns the histograms for probabilistic scoring:

```python
dist = model.predict_distribution(X_test)      # one bagged fit, like predict()
point = dist.point_prediction                  # the predict() output of this fit
crps = dist.crps(y_test).mean()                # exact CRPS, target units
log_score = -dist.log_prob(y_test).mean()      # log_prob is -inf where the density is 0
q10, q50, q90 = dist.quantile([0.1, 0.5, 0.9]).T
edges, probs = dist.bin_edges, dist.probabilities   # raw per-child histograms
```

`bin_edges` has shape `(n_children, 1001)` and `probabilities`
`(n_children, n_test, 1000)`, in target units. Each child bins the target on a
fixed grid in its own normalized space (`linspace(-0.5, 1.5, 1001)` over the
training-fold range), so the children's grids can differ, and the bagged
predictive distribution is the equal-weight mixture of their histograms.
`RegressionDistribution` evaluates that mixture exactly, treating each bin's
mass as uniform within the bin: `mean`, `cdf`, `pdf`, `log_prob`, `quantile`,
`crps`. By default the histograms are captured from the same forward pass that
produces the point predictions, so `dist.mean` reproduces `point_prediction`
up to float32 rounding; with `MITRA_HELDOUT_IN_SUPPORT=0` they come from one
extra forward-only pass over `X_test` and can differ above the predict-time
support cap (the gap is logged). Memory is about 32 KB per test row with eight
children.

## Speed (on by default)

Version 0.3 makes the fine-tuning loop and the prediction cheaper without
changing what a step computes or, in expectation, what is predicted. Every
item below is on by default and can be switched off with an environment
variable, read at fit time.

### Fine-tuning loop

Four changes to the loop, none to the recipe (`speed.py`, applied through the
package's trainer subclass):

- The validation pass after every step predicts the validation set in one
  wide query chunk (16,384 rows) instead of stock's 1,024-row chunks with a
  fresh support draw each; on large tables that pass cost as much as several
  steps. The transformed arrays it scores are computed once per fit instead
  of once per step.
- Before the first validation pass, one throw-away forward and backward pass
  at the fine-tuning context size makes a context that does not fit the GPU
  fail in seconds instead of after a full validation pass; AutoGluon's
  out-of-memory ratchet then halves the context as before. Later bag children
  start at the context that fit the first child instead of repeating its
  failed attempts.
- The loop's attention runs on torch's `scaled_dot_product_attention`, which
  matches flash-attn 2 to bf16 rounding, is as fast per step on an H100 and
  1.3 to 1.7 times faster on an RTX PRO 6000 Blackwell. Prediction keeps the
  construction-time kernel.
- The best-weights checkpoint stays on the GPU instead of being copied to the
  host at every improving step, and validation runs under `inference_mode`.

On the largest TabArena tables a step costs about half of what it did, so
about twice as many of the 50 steps fit a 250 s budget (RTX PRO 6000:
Diabetes130US 11 to 26 steps, APSFailure 15 to 30, kddcup09_appetency 15 to
26); tables that already reached 50 steps simply finish sooner.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MITRA_FT_FAST_LOOP` | `1` | `0` restores the stock loop (all four items off) |
| `MITRA_FT_ATTENTION` | `sdpa` | `stock` keeps the construction-time attention kernel in the loop |
| `MITRA_FT_EVAL_CHUNK` | `16384` | query rows per validation chunk (halved under out-of-memory, down to the stock 1,024) |
| `MITRA_FT_PREFLIGHT` | `1` | `0` skips the memory preflight |
| `MITRA_FITTED_CONTEXT_MEMO` | `1` | `0` makes every bag child repeat the out-of-memory ratchet |

### Prediction

Stock AutoGluon Mitra predicts in 1,024-row query chunks and re-draws the
in-context support for every chunk, so a large test set re-encodes the
support many times. This package predicts in 16,384-row chunks whether or
not the support fits its cap. When it fits (the common case) the support is
drawn once per test set, so a single-chunk predict is bit-identical to stock.
When the training table exceeds the cap, every chunk still gets a fresh
capped draw, as in stock, so every query row sees exactly one draw and the
prediction is the same in expectation; measured on the ten TabArena tables
in that regime (90 splits, paired against the stock chunking on the same
GPUs): 50 wins, 40 losses, geometric error ratio 1.0001, and 7 times less
inference time (APSFailure 1,419 s to 185 s per split). Over the whole
51-dataset suite the total prediction time is 5.2 times lower.

Fine-tuning is untouched by this; only the forward-only prediction path
changes. On a prediction-time CUDA out-of-memory the query chunk is shrunk
first (a pure batching change): down to `MITRA_FAST_PREDICT_QCHUNK_FLOOR`
when the support fits, and to the stock 1,024 rows when it is capped, since
below that the prediction would only get slower, never different. Only then
is the support cap halved.

To restore the stock per-chunk behavior, set `MITRA_FAST_PREDICT=0`. The
relevant environment variables (defaults shown):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MITRA_FAST_PREDICT` | `1` | `0` disables the speedup (stock 1,024-row chunks with a redraw each) |
| `MITRA_FAST_PREDICT_QCHUNK` | `16384` | query rows per prediction chunk |
| `MITRA_FAST_PREDICT_QCHUNK_FLOOR` | `256` | smallest chunk the OOM fallback will use when the support fits |

### Further speedup for many-chunk prediction: the support cache (opt-in)

The fast-prediction lever above makes most tables single-pass; test sets
that still need several query chunks (very large tests, or chunks shrunk
by the OOM fallback) continue to re-encode the support once per chunk.
Setting

```bash
export MITRA_SUPPORT_CACHE=1        # off by default
export MITRA_SUPPORT_CACHE_GB=6     # cache memory budget (falls back if exceeded)
```

encodes the support once per predict call and reuses it for every chunk;
the support stream never attends to the query, so this is mathematically
exact. Measured (H100, an earlier Mitra checkpoint): ~5x faster prediction
(bit-identical to the uncached standard-attention path), 1.25x end-to-end
on a large benchmark task (fine-tuning itself is unchanged; the cache
only accelerates prediction after the weights are frozen). The win scales
with the number of query chunks, so it is largest for big test sets and
for serving many predictions from one fitted model.

Two caveats: on flash-attn installs the cached path computes prediction
with standard attention (different kernel, same math; observed effect on
task metrics ~1e-5), and on tasks with more training rows than the support
cap it fixes one support subsample per predict call instead of redrawing
per chunk.

## Notes

- **Protocol parity:** the fit is an AutoGluon 8-fold bagged fine-tune:
  eight child models whose probabilities are averaged, with out-of-fold
  validation over the full training set. This is exactly the protocol
  behind the reported benchmark numbers.
- The fit runs in a subprocess because the fine-tuning controls patch
  AutoGluon's Mitra internals process-globally.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Evaluation results

Our TabArena evaluation results for the released checkpoints can be found in [results/](results/).

## Reference

[Mitra-v2 Technical Report](https://arxiv.org/abs/2609.04540) (Amazon, 2026), also available on the [Hub](https://huggingface.co/autogluon/mitra-finetune/blob/main/Mitra_v2_Technical_Report.pdf).

```
@article{mitrav2_2026,
  title={{Mitra-v2} Technical Report},
  author={Tao, Yefan and Zhang, Xiyuan and Liu, Xinyi and Han, Boran and Maddix, Danielle and Fang, Haoyang and Han, Zhen and Gai, Jiading and Liu, Xuanqing and Bohlke-Schneider, Michael and Wang, Yuyang (Bernie) and Friedland, Gerald and Mah, Kevan and Lee, Chris and Kong, Chris},
  journal={arXiv preprint arXiv:2609.04540},
  year={2026}
}
```
