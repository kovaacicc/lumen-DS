# Codebase Audit — Lumen DS Competition Project

> Generated 2026-05-07. Last updated 2026-05-07 — cleanup applied to `data_cleaning`, `market_basket`, `customer_segmentation`. See `CLEANUP_REMAINING.md` for pending notebooks.

---

## Notebook Summary

### 1. `notebooks/data_cleaning.ipynb`
**Purpose:** Full EDA on the raw dataset followed by a two-phase cleaning pipeline: first removes problematic rows (SAMPLE/slash item codes, negative prices, negative quantities, garbage cost-of-part records), then imputes remaining nulls (Unknown-fill for categoricals, GM% recalculation where price and cost are both available). Produces the canonical clean dataset used by every downstream notebook.

**Inputs:**
- `data/raw/LUMEN_DS.csv` (via `dataprep/fixer.py` → `data/raw/data.csv` → `dataprep/cleaner.py`)
- Re-loads `data/processed/data_clean_v1.csv` mid-notebook for the imputation phase

**Outputs:**
- `data/processed/data_clean_v1.csv` (after row filtering)
- `data/processed/data_clean_final.csv` (after imputation — primary output)

**Status:** Complete. 66 cells, all executed with output. Three intermediate blocks are disabled via `"""..."""` string wrapping (not actual comments).

---

### 2. `notebooks/encoding_and_feature_engineering.ipynb`
**Purpose:** Encodes categorical variables (label encoding for high-cardinality columns, one-hot for low-cardinality), engineers derived features (customer tenure, line revenue, gross profit, temporal features), builds customer-level aggregates for segmentation, and computes a product co-occurrence table for market basket analysis.

**Inputs:**
- `data/processed/data_clean_final.csv`

**Outputs:**
- `data/processed/data_clean_enc_fe.csv` (928k rows × 62 cols — generated but gitignored, never read downstream)
- `data/processed/customer_features.csv` (11,753 customers × 12 cols)
- `data/processed/invoice_items.csv` (invoice-level product lists — generated but gitignored, never read downstream)

**Status:** Complete. All cells executed.

---

### 3. `notebooks/customer_segmentation.ipynb`
**Purpose:** RFM scoring (Recency, Frequency, Monetary) with rule-based segment labels, followed by K-Means clustering (K=4, ROT from elbow/silhouette analysis) that assigns each customer to one of four behavioral clusters (Whales/Champions, Active Small Buyers, Mid-Tier At Risk, Hibernating). Produces the `Cluster_name` feature used by recommender and upsell models.

**Inputs:**
- `data/processed/data_clean_final.csv`

**Outputs:**
- `data/processed/rfm_segments.csv` (11,753 customers, 14 columns incl. K3, K4, Cluster, Cluster_name, RFM scores)
- `visualization/clusters_3d.html` (interactive Plotly scatter)

**Status:** Complete. All cells executed.

---

### 4. `notebooks/market_basket.ipynb`
**Purpose:** Derives cross-sell association rules using Apriori and FP-Growth at the customer-monthly basket level (not invoice level, with justification). Includes a theoretical introduction, sensitivity analysis for `min_support`, verification that both algorithms produce identical results, and a bonus Item Code–level analysis concluding that Product Group is the correct granularity. Saves rules and basket matrix for use by the recommender.

**Inputs:**
- `data/processed/data_clean_final.csv`

**Outputs:**
- `data/processed/association_rules.csv` (14 rules: lift > 2.0, confidence > 0.30, conviction > 1.3)
- `data/processed/basket_matrix.csv` (89,388 × 31 — gitignored)

**Status:** Complete. All cells executed. Two blocks are disabled as string literals: the first pass at rule generation (min\_lift=1.0, min\_confidence=0.25) and the full per-customer recommendation generation loop (both replaced or noted as out-of-scope).

---

### 5. `notebooks/cross_sell_model.ipynb`
**Purpose:** An earlier binary classifier that predicts whether a customer will ever buy from multiple product groups, treating it as a customer-level yes/no classification problem. Trains Random Forest (AUC 0.91) and XGBoost (AUC 0.92). **This notebook is superseded by `cross_sell_recommender.ipynb`** — it does not produce item-level recommendations and its models are not loaded by the API.

**Inputs:**
- `data/processed/customer_features.csv`

**Outputs:**
- `data/processed/cross_sell/customer_predictions_rf.csv` (gitignored directory)
- `data/processed/cross_sell/customer_predictions_xgb.csv` (gitignored directory)
- `models/cross_sell/cross_sell_rf.pkl`
- `models/cross_sell/cross_sell_xgb.pkl`

**Status:** Complete. All cells executed. Models are saved but not used by the API.

---

### 6. `notebooks/cross_sell_recommender.ipynb`
**Purpose:** The production cross-sell recommender. Builds a Customer × Product Group matrix (unseen PGs only), engineers 27 features per (customer, PG) pair from three sources — customer RFM/history, product group statistics, and association rule / co-purchase signal — then trains Logistic Regression, Random Forest, and XGBoost. XGBoost wins (AUC 0.9248, 5-fold CV 0.9291 ± 0.0055). Also tries rolling window training (3 historical windows; did not improve over base). Saves all inference artifacts for the API.

**Inputs:**
- `data/processed/data_clean_final.csv`
- `data/processed/rfm_segments.csv`
- `data/processed/association_rules.csv`

**Outputs:**
- `data/processed/recommender/customer_feats.csv`
- `data/processed/recommender/customer_purchased_pgs.csv`
- `data/processed/recommender/product_feats.csv`
- `data/processed/recommender/copurchase_rates.csv`
- `data/processed/recommender/feature_medians.json`
- `data/processed/recommender/customer_recommendations.csv`
- `models/recommender/recommender_best.pkl` (XGBoost)
- `models/recommender/recommender_rf.pkl`
- `models/recommender/recommender_xgb.pkl`
- `models/recommender/recommender_scaler.pkl`
- `models/recommender/model_meta.json`

**Status:** Complete. All cells executed.

---

### 7. `notebooks/upsell_model.ipynb`
**Purpose:** Identifies up-sell targets (customers buying below the median price in a product group) and recommends premium-tier items popular within the customer's RFM cluster. Generates 27,744 recommendations covering 4,843 customers across 26 product groups. Validates with a label-window proxy: 44.2% of targets show positive price uplift in the holdout period, 7.6% reached premium tier.

**Inputs:**
- `data/processed/data_clean_final.csv`
- `data/processed/rfm_segments.csv`

**Outputs:**
- `data/processed/upsell/upsell_recommendations.csv`
- `data/processed/upsell/item_tiers.csv`
- `data/processed/upsell/upsell_item_popularity.csv` (premium items only)
- `data/processed/upsell/upsell_item_popularity_all.csv` (all items — Layer 2 for cross-sell API)
- `data/processed/upsell/pg_median_prices.csv`
- `models/upsell/upsell_meta.json`
- `visualization/upsell_validation.png`

**Status:** Complete. All cells executed.

---

### Helper Scripts

**`dataprep/fixer.py`** — Reads `../LUMEN_DS.csv` (UTF-16), strips double quotes, writes `../data.csv` (UTF-8). Hardcoded paths relative to the `dataprep/` directory. 7 lines.

**`dataprep/cleaner.py`** — Reads `../data.csv` (pipe-delimited), drops 12 columns, drops rows with null key identifiers, fills categorical nulls with `"unknown"`. Hardcoded paths relative to `dataprep/`. 49 lines. Note: `data_cleaning.ipynb` re-implements this logic inline and does not import or call this file.

**`api/main.py`** — FastAPI application with `/health`, `/cross-sell`, and `/up-sell` endpoints. Loads all models and lookup tables once at startup via `lifespan`. Cross-sell uses the XGBoost recommender; up-sell uses cluster-conditional item popularity tables. ~328 lines. (Reportedly still being worked on.)

---

## Pipeline Flow

```
data/raw/LUMEN_DS.csv  (UTF-16, pipe-sep — gitignored)
    │
    └─ dataprep/fixer.py → data/raw/data.csv  (UTF-8 — gitignored)
         │
         └─ dataprep/cleaner.py (drops 12 cols, fills nulls)
              │
              └─ [01] data_cleaning.ipynb
                   │  (EDA, row filtering, imputation)
                   │
                   ├──→ data/processed/data_clean_v1.csv  (gitignored)
                   └──→ data/processed/data_clean_final.csv  (gitignored)
                              │
              ┌───────────────┼──────────────────────┐
              │               │                      │
              ▼               ▼                      ▼
   [02] encoding_fe.ipynb  [02] customer_seg.ipynb  [03] market_basket.ipynb
              │               │                      │
              ▼               ▼                      ▼
   customer_features.csv   rfm_segments.csv     association_rules.csv
   data_clean_enc_fe.csv*                       basket_matrix.csv*
   invoice_items.csv*
              │               │                      │
              │               └──────────┬───────────┘
              │                          │
              ▼                          ▼
   [04a] cross_sell_model.ipynb   [04b] cross_sell_recommender.ipynb
   (superseded, not used by API)   (production recommender)
              │                          │
              ▼                          ▼
   models/cross_sell/*.pkl         models/recommender/*.pkl
   (not loaded by API)             data/processed/recommender/*.csv
                                          │
                                          ▼
                                   [05] upsell_model.ipynb
                                          │
                                          ▼
                                   data/processed/upsell/*.csv
                                   models/upsell/upsell_meta.json
                                          │
                              ┌───────────┴──────────────────┐
                              │        api/main.py            │
                              │  /cross-sell  /up-sell       │
                              └──────────────────────────────┘

* = gitignored, generated locally only
```

**Execution order:** `data_cleaning` → `encoding_and_feature_engineering` → `customer_segmentation` → `market_basket` → `cross_sell_recommender` + `upsell_model` (can run in parallel) → API

---

## Cleanup Needed

### `data_cleaning.ipynb` ✅ cleaned 2026-05-07
- ~~**Dead code (cells 5 & 12):** first `already_dropped` definition~~ — **deleted**
- ~~**Disabled blocks via string literals:** cells 53, 56, 58~~ — **deleted** (TX imputation, cost imputation, GM% estimation)
- ~~**Redundant exploratory displays:** cells 15–18~~ — **deleted** (Sales Channel + Manufacturing EDA); Intercompany filter kept, EDA prints removed
- ~~**Cell 44:** `data1.csv` debugging remnant~~ — **deleted**
- ~~**Cell 28:** `surviving_ss` sanity check~~ — **deleted**
- **Title cell** — updated with English purpose description
- 66 cells → 57 cells

### `encoding_and_feature_engineering.ipynb` ⏳ pending — see `CLEANUP_REMAINING.md`
- **Dead code (cell `940b426b`):** Entire `Customer industry` numeric extraction block is commented out. Should be deleted.
- **`Intercompany` column:** All 928,762 rows have value `"NO"` (printed in cell output). Including it in encodings adds zero signal; encoding it produces a constant column.
- **Orphaned outputs:** `data_clean_enc_fe.csv` (~928k × 62 cols, gitignored) and `invoice_items.csv` (gitignored) are saved but never read by any downstream notebook — both can be removed from the save block.
- **Empty trailing cell** (`e31f8003`): blank cell at the end.

### `customer_segmentation.ipynb` ✅ cleaned 2026-05-07
- ~~**Unused K3 columns:** `for k in [3, 4]` loop~~ — **deleted** (K=4 is recomputed in the final cell anyway)
- ~~**`import re` inside function:**~~ — **moved** to top-level imports
- ~~**Standalone `%matplotlib inline` cell** (`d639e1f8`)~~ — **deleted**
- **Title markdown cell** — **added** at position 0
- 16 cells → 15 cells

### `market_basket.ipynb` ✅ cleaned 2026-05-07
- ~~**Disabled block `cell-rules`** (lax thresholds, min\_lift=1.0)~~ — **deleted**
- ~~**Disabled block `cell-all-recommendations`** (all-customer loop)~~ — **deleted**
- ~~**`customer_crosssell.to_csv(...)` string literal in save cell**~~ — **removed**
- ~~**`ext = df` alias** (`f2a09315`)~~ — **removed**; all `ext[...]` → `df[...]`
- ~~**Croatian inline comment** (`13881fa0`)~~ — **translated** to English
- 22 cells → 20 cells

### `cross_sell_model.ipynb` ⏳ pending — see `CLEANUP_REMAINING.md`
- **Entire notebook is superseded.** The API does not load any model from `models/cross_sell/`. The notebook should be either clearly marked as "exploratory/archived" in a header markdown cell, or moved to a separate `archived/` directory to avoid confusion.
- **No temporal train/test split:** The model is trained on all historical data without leakage prevention — invalid for rigorous evaluation of a time-series prediction task. The notebook's stated purpose (identify multi-PG buyers) is a static classification, not a prospective recommendation.
- **`scale_pos_weight` hardcoded:** `scale_pos_weight=7220/4554` is hardcoded from a data snapshot. This will silently become wrong if the dataset changes.

### `cross_sell_recommender.ipynb` ⏳ pending — see `CLEANUP_REMAINING.md`
- **`BASELINE_XGB_AUC = 0.8888` hardcoded** at the start of the rolling window section — this documents a prior state but will silently be wrong if the model is retrained. Convert to a comment or compute it live.
- **Rolling window `LabelEncoder` re-fit:** In `build_window_df()`, `le_w = LabelEncoder()` is fit on each window's data independently. If a window's `customer_region` column has fewer unique values than the main train set, the integer encodings will differ from the main model's `cluster_enc`, `region_enc`, etc. — a silent feature mismatch.
- **`model_df['xgb_probability']` written in the recommendations cell** then `model_df['best_probability']` written again in the save cell — two columns with near-identical content in the large in-memory DataFrame, unnecessary.
- **Precision@5 = 0.38%** is noted but not contextualized. At 290:1 class imbalance, a random ranker would score ~0.34% — the model barely exceeds random on this metric. Worth a brief markdown note.

### `upsell_model.ipynb` ⏳ pending — see `CLEANUP_REMAINING.md`
- **`SF001` edge case:** `SF001` has only 1 unique item (`363999`) which achieves 100% confidence with 2 buyers — technically valid but likely a degenerate recommendation for any customer in that group.
- **Validation section lacks interpretation:** 7.6% of targets moved to premium tier is stated without a baseline (what % would move in the absence of targeting?). A single-line comparison to a random-customer baseline would strengthen the section.
- **`n_buyers >= 2` filter comment** is present but the rationale ("removes single-purchase noise") could be a markdown cell rather than an inline comment, since it's a significant methodological choice.

### `api/main.py`
- **`state: dict = {}` global mutable state:** Works for single-process FastAPI, but any future multi-worker deployment (Gunicorn with multiple workers) will not share this state. Not an immediate bug, but worth noting.
- **`Cluster_name` lookup in `/cross-sell`:** `cust.get("Cluster_name")` works because the raw CSV has that column, but `cluster_enc` (used in model features) is a different integer. If the CSV is re-generated with different cluster assignments, the encoded value and the string name may disagree.
- **No authentication or rate limiting** on either endpoint.
- **`_load_data()` rule\_lookup construction** iterates over all customers × all rules at startup — scales as O(customers × rules). With 9,017 customers and 14 rules this is fine; worth a note if rules grow.
- **`requirements.txt` missing version pins** for `pandas`, `numpy`, `scikit-learn` — risks silent breakage across `pandas` major versions (e.g., 1.x → 2.x changed `DataFrame.apply` group behavior, which shows up as a `FutureWarning` in market\_basket.ipynb cell `f2a09315`).
- **`imbalanced-learn` not in `requirements.txt`** — it is only used in notebooks (not the API itself), so the API requirements are correct, but a top-level `requirements-notebooks.txt` does not exist.

---

## Documentation Gaps

1. **No README.md.** There is no top-level file explaining what the project does, how to set up the environment, what the raw data looks like, or how to run the pipeline. A newcomer has no entry point.

2. **Execution order not documented.** Notebooks must be run in a specific sequence (`data_cleaning` → `encoding_and_feature_engineering` → `customer_segmentation` → `market_basket` → `cross_sell_recommender` + `upsell_model`). This dependency chain is invisible without reading all notebooks.

3. **`cross_sell_model.ipynb` vs `cross_sell_recommender.ipynb` ambiguity.** These two notebooks address similar problems with different approaches. The former is not used by the API, but there is no comment or header explaining this. A reviewer would reasonably assume both are production code.

4. **Raw data is gitignored with no instructions.** `data/raw/` is entirely excluded. Someone cloning the repo gets no guidance on where to obtain the original dataset, what format it is in (UTF-16, pipe-separated CSV), or what schema it has beyond the brief `opisi/data_explanation.txt` (written in Croatian).

5. **`data_clean_final.csv` is gitignored** (matched by `data/processed/data_clean*.csv`), meaning the most critical intermediate artifact — required by 5 of 7 notebooks — is not committed. Any collaborator must re-run `data_cleaning.ipynb` from scratch before running anything else, but this is not stated anywhere.

6. **`data_clean_enc_fe.csv` is generated but never consumed.** Encoding notebook saves a 928k-row encoded dataset that no subsequent notebook reads. Every downstream notebook re-reads `data_clean_final.csv` and reconstructs features internally. The file wastes disk space and creates a false expectation that it is an important pipeline output.

7. **`invoice_items.csv` is generated but never consumed** for the same reason — `cross_sell_recommender.ipynb` builds its own invoice-level tables from scratch.

8. **`dataprep/cleaner.py` and `data_cleaning.ipynb` are not connected.** The script is standalone with hardcoded `../data.csv` paths (relative to `dataprep/`). The notebook re-implements the same logic inline. There is no import, no `subprocess` call, no `%run`. The script appears to be a first-pass attempt that was superseded by the notebook without being removed or documented.

9. **No model versioning or reproducibility hash.** `model_meta.json` records AUC scores and hyperparameters but not a timestamp or data hash. If `.pkl` files are regenerated (e.g., retraining on new data), there is no way to detect that the metadata is stale.

10. **API startup behavior undocumented.** `api/requirements.txt` exists but there is no instruction on how to start the API (`uvicorn api.main:app`), what port it runs on, or how to test the endpoints. No `Makefile`, no `docker-compose`, no startup script.

11. **Croatian language barrier.** Most comments, variable names for intermediate results, and all of `opisi/` are in Croatian. The code itself is English, but the explanatory layer is inaccessible to non-Croatian reviewers without translation.

12. **Validation methodology not fully justified.** `cross_sell_recommender.ipynb` reports Precision@5 = 0.38% against a 0.34% random baseline — marginal improvement. `upsell_model.ipynb` reports 7.6% premium migration without a control group baseline. Neither notebook contains a discussion of whether these metrics are sufficient for the competition's evaluation criteria.

13. **`data/processed/data_clean_final_sample.csv` has no known origin.** This file is present on disk but is not generated by any notebook in the repo. It is likely a manually created sample used for quick testing, but it is undocumented and its relationship to the main pipeline is unclear.
