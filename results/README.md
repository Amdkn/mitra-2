# Mitra-v2 evaluation results

Per-unit TabArena results for the released Mitra-v2 checkpoints under the
TabArena one-hour protocol (3,600 s per-unit limit, 250 s per-bag-child
fine-tuning budget, eight-fold bagging, heldout-in-support), as reported in the
Mitra-v2 technical report. The format matches the per-unit result files that
TabFM publishes: one row per dataset and fold.

| File | Rows | Datasets | Metric |
|---|---|---|---|
| `mitra-v2-tabarena-classification.parquet` | 594 | 38 | roc_auc (binary), log_loss (multiclass) |
| `mitra-v2-tabarena-regression.parquet` | 222 | 13 | rmse |

Columns: `dataset`, `fold` (0-29 = TabArena repeat x fold index), `method`
(`Mitra-v2`), `metric_error`, `metric`, `problem_type`.

```python
import pandas as pd
cls = pd.read_parquet("results/mitra-v2-tabarena-classification.parquet")
reg = pd.read_parquet("results/mitra-v2-tabarena-regression.parquet")
```
