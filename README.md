# smartstore-ml

Labeled sample cases and a training script for the
`RandomForestClassifier` that `smartstore.py`'s `RecommendationEngine`
already imports but never trains or uses.

## Layout

```
smartstore-ml/
├── data/
│   └── training_cases.csv     # 60 labeled example files
├── model/
│   ├── cleanup_model.joblib       # fitted RandomForestClassifier
│   ├── feature_columns.json       # exact column order the model expects
│   └── extension_categories.json  # extension -> category groupings
├── train_model.py
└── README.md
```

## `training_cases.csv` schema

One row per example file, mirroring the fields already computed by
`FileScanner` in `smartstore.py`:

| column | meaning |
|---|---|
| `path_example` | illustrative path only, not used as a feature |
| `size_mb` | file size in MB |
| `age_days` | days since last modified |
| `unused_days` | days since last accessed |
| `extension` | file extension including the dot, blank if none |
| `is_cache` / `is_temp` / `is_log` / `is_hidden` | 0/1 flags, same definitions as `FileRecord` |
| `is_protected` | 0/1, matches `SafetyEngine.is_protected` |
| `is_user_data` | 0/1, matches `SafetyEngine.is_user_data` |
| `label` | ground truth: `KEEP`, `REVIEW`, or `CLEAN` |

The 60 rows cover the scenarios the rule-based scorer already
special-cases (cache dirs, temp/swap/backup files, rotated logs,
package caches) plus edge cases it's shakier on: large disk images,
old-but-important documents, protected system paths, git pack files
inside a "Projects" folder, and files that are borderline based on
combinations of signals rather than any single flag.

**This dataset is illustrative, not exhaustive.** Before trusting a
model trained on it, add real cases from actual scans (see below).

## Training

```bash
cd smartstore-ml
python3 train_model.py
```

This prints a held-out classification report and feature
importances, then refits on the full dataset and saves:

- `model/cleanup_model.joblib`
- `model/feature_columns.json`
- `model/extension_categories.json`

Flags: `--data <csv>`, `--out <dir>`, `--test-size 0.25`, `--seed 42`.

## Adding more cases

The fastest way to grow this dataset honestly is to pull real
examples from your own `~/.smartstore/smartstore.db` `actions` and
`feedback` tables (already logged by `smartstore.py`) once you've
run `clean` / `undo` a few times — those rows have a real user
verdict attached, which is worth more than hand-written examples.
Until then, append rows to `training_cases.csv` by hand, keeping the
same columns, and re-run `train_model.py`.

60 rows is enough to get the pipeline working end-to-end; it is
**not** enough to trust the model's predictions in place of the
rule-based `RecommendationEngine.calculate_score`. Treat this as
scaffolding to grow, not a finished classifier.

## Wiring it into `smartstore.py` (not yet done)

`RecommendationEngine.__init__` already tries to construct a
`RandomForestClassifier` if scikit-learn is installed, but nothing
in `recommend()` currently calls `self.ml_model.predict(...)` — the
rule-based score is the only thing driving output today. To actually
use the trained model, `recommend()` would need to:

1. Build the same feature vector this script builds (via
   `feature_columns.json` for column order and
   `extension_categories.json` for the extension grouping)
2. Call `self.ml_model.predict_proba(...)` if `model/cleanup_model.joblib`
   exists on disk
3. Decide how to combine that prediction with `SafetyEngine`'s
   `is_protected` / `is_user_data` checks — those should still hard-override
   the model, the same way they hard-override the rule-based score today
