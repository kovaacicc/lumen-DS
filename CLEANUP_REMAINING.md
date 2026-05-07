# Cleanup Guide — Remaining Notebooks

Follow the same rules that were applied to `data_cleaning`, `market_basket`, and `customer_segmentation`:
- Remove dead code cells, disabled blocks, unused imports
- Standardize markdown: title cell at top, `## N.` section headers, brief purpose description
- Do NOT change any logic, variable names, or model parameters
- Do NOT remove output cells that show results/charts

---

## `encoding_and_feature_engineering.ipynb`

### 1. Add title markdown cell at position 0 (before existing cell `76979871`)

Insert a new **markdown** cell at the very top. Suggested content:

```markdown
# 03 — Encoding & Feature Engineering
**Lumen Data Science — Cross-Sell & Up-Sell Prediction**

Encodes categorical variables, engineers derived features (tenure, line revenue, GM%), and
builds customer-level aggregate statistics used by `cross_sell_model.ipynb`.

**Input:** `data/processed/data_clean_final.csv`
**Output:** `data/processed/customer_features.csv` — customer aggregates consumed by cross_sell_model
```

Note: `data_clean_enc_fe.csv` and `invoice_items.csv` are also saved here but are not read by any other notebook.

---

### 2. Delete cell `940b426b` — entire commented-out Customer industry block

The cell contains:
```python
# # Customer industry enkodiranje (IC000, IC001... → numerički)
# # Izvuci broj iz stringa
# if 'Customer industry' in df_encoded.columns:
#     ...
```
Every line is commented out with `#`. This is dead code — delete the whole cell.

---

### 3. Edit cell `f536ddc7` — remove orphaned output saves

Current save cell writes three files. Only `customer_features.csv` is ever read downstream.
`data_clean_enc_fe.csv` (928k × 62 cols) and `invoice_items.csv` are generated but never consumed
by any notebook and are gitignored.

**Change this:**
```python
df_encoded.to_csv('../data/processed/data_clean_enc_fe.csv', index=False)
customer_agg.to_csv('../data/processed/customer_features.csv', index=False)
invoice_items.to_csv('../data/processed/invoice_items.csv', index=False)

print('Spremljeno ✓')
print('  → data/processed/data_clean_enc_fe.csv')
print('  → data/processed/customer_features.csv')
print('  → data/processed/invoice_items.csv')
```

**To this:**
```python
customer_agg.to_csv('../data/processed/customer_features.csv', index=False)
print('Saved: data/processed/customer_features.csv')
```

---

### 4. Delete cell `e31f8003` — empty trailing cell

The last cell in the notebook is completely empty. Delete it.

---

## `cross_sell_model.ipynb`

This notebook is **superseded** by `cross_sell_recommender.ipynb`. The models it saves
(`models/cross_sell/cross_sell_rf.pkl`, `cross_sell_xgb.pkl`) are not loaded by the API.
The only change needed is making that clear so reviewers aren't confused.

### 1. Edit cell `md-intro` — mark as archived

**Replace** the current title cell content with:

```markdown
# Cross-Sell Model — EXPLORATORY / ARCHIVED

> **This notebook is not part of the production pipeline.**
> The API uses `cross_sell_recommender.ipynb` (XGBoost, AUC 0.9248) instead.
> This notebook predicts *whether* a customer will buy from multiple product groups
> (binary yes/no), not *which* product groups to recommend. It is kept for reference.

**Limitations vs the production recommender:**
- No temporal train/test split (no leakage protection)
- No item-level recommendations — output is a customer-level flag
- `scale_pos_weight=7220/4554` is hardcoded from one data snapshot

**Input:** `data/processed/customer_features.csv`
**Output:** `models/cross_sell/` — not loaded by the API
```

No other changes needed in this notebook.

---

## `cross_sell_recommender.ipynb`

### 1. Edit cell `9547479f` — convert hardcoded baseline to a comment

Current first line:
```python
BASELINE_XGB_AUC = 0.8888  # XGBoost AUC before co-purchase features (original baseline)
```

This will silently be wrong if the model is retrained. **Replace** with a comment:
```python
# Baseline XGB AUC before co-purchase features was 0.8888 (logged at time of development).
# Retraining will change this — do not use as a live comparison value.
```

---

### 2. Edit markdown cell `md-section10` — add Precision@5 context

Current content ends with the section header. **Append** this note below the existing text:

```markdown
> **Context:** At a 290:1 class imbalance, a random ranker would score ≈ 0.34% Precision@5.
> The model achieves 0.38% — meaningful improvement but modest in absolute terms.
> AUC (0.9248) is the primary metric; Precision@K is included as a sanity check.
```

---

### 3. Note about `cell-recommendat` (cell index 36) — `xgb_probability` column

Cell `cell-recommendat` adds `model_df['xgb_probability']` and builds a `recommendations`
DataFrame from it. Cell `cell-save` (index 38) later adds `model_df['best_probability']` and
builds the *actual* saved recommendations.

This means `recommendations` is defined twice and `xgb_probability` is a dead column in memory.
**Optional cleanup:** delete `cell-recommendat` entirely — the final recommendations are produced
in `cell-save`. If you keep it as a preview step, rename the variable to `recommendations_preview`
so the intent is clear.

---

## `upsell_model.ipynb`

### 1. Convert the `n_buyers >= 2` inline comment to a markdown cell

In cell `9e19bb4b`, just before the line:
```python
item_popularity = item_popularity[item_popularity['n_buyers'] >= 2]
```
there is an inline comment: `# Remove single-buyer noise: 1/1 = 100% confidence from one purchase is meaningless`.

This is a significant methodological choice. **Insert a markdown cell above `9e19bb4b`:**

```markdown
### Filtering: minimum 2 buyers per (Cluster, PG, Item)

Items with only 1 buyer in a cluster–PG group get 100% confidence from a single purchase,
which is noise rather than signal. The `n_buyers >= 2` filter removes these degenerate cases.

**Known edge case — SF001 (Fittings):** this PG has only one item (`363999`), so any
cluster with 2+ buyers of that item still hits 100% confidence. Technically valid,
but the recommendation is trivially "buy the only item in this group."
```

---

### 2. Edit cell `32e9e7aa` (Validation section header) — add baseline note

The validation section reports "44.2% positive price uplift, 7.6% moved to premium tier" but
gives no baseline. **Add the following line at the end of the markdown cell:**

```markdown
> **Missing baseline:** without a control group we cannot separate model effect from natural
> price drift. A rough sanity check: compute the same premium-migration rate on a random
> sample of non-targeted customers in the same PGs during the same label window. If both
> rates are similar, the 7.6% figure reflects market trend rather than model quality.
```

---

*These notebooks were not cleaned alongside `data_cleaning`, `market_basket`, and `customer_segmentation`
because they were authored by a different team member. Apply the same judgment: do not change logic,
variable names, or model parameters — only remove dead code and improve documentation.*
