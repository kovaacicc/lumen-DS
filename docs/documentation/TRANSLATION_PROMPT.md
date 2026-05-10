# Croatian Translation Prompt

> Paste the prompt below into a fresh Claude Code instance running in this same project (`/home/karlo/projects/ml/lumen`). It will produce a Croatian version of `DOCUMENTATION.md` saved to `docs/documentation/DOKUMENTACIJA.md`.

---

## Prompt to paste

You are translating the project's CRISP-DM documentation from English into Croatian.

**Source file:** `docs/documentation/DOCUMENTATION.md`
**Target file (create new):** `docs/documentation/DOKUMENTACIJA.md`

Read the source file in full, then produce a complete Croatian translation following the rules below. Save the result with the Write tool — do not modify the English source.

### Translation rules

1. **Translate prose, headings, table cells, and captions** into natural, technically fluent Croatian. Use the tone of a professional data-science report, not a literal word-for-word translation.

2. **Do NOT translate:**
   - Code blocks, code snippets, or any text inside backticks (`like_this`).
   - Column names from the dataset (`Invoice Date`, `CustomerID`, `Item Code`, `Product group`, `GM%`, `Invoiced price`, `Invoiced qty (shipped)`, `Cost of part`, `Customer industry`, `Customer Region`, `Top Customer Group`, `Sales Channel - Internal/External`, `Intercompany`, `Make vs Buy`, `Manufacturing Region`, `Manufacturing Location Code`, `Customer First Invoice Date`, `Born on date`, `Price last modified date in the ERP`, `Order #`, `Order Date`, `Order Line #`, `Invoice #`, `Invoice Line #`, `Product family`, `Ordered qty`, `Material cost`, `Labor cost`, `Overhead cost`).
   - Cluster names: `Whales / Champions`, `Active Small Buyers`, `Mid-Tier At Risk`, `Hibernating`, `Cluster_name`.
   - Algorithm and library names: `Apriori`, `FP-Growth`, `K-Means`, `XGBoost`, `Random Forest`, `Logistic Regression`, `SMOTE`, `GroupKFold`, `StandardScaler`, `LabelEncoder`, `RFM`, `FastAPI`, `uvicorn`, `joblib`, `scikit-learn`, `imbalanced-learn`, `mlxtend`, `pandas`, `numpy`.
   - Metric names: `ROC-AUC`, `AUC`, `Precision@K`, `Precision@5`, `Recency`, `Frequency`, `Monetary`, `support`, `confidence`, `lift`, `conviction`, `silhouette score`.
   - File paths, URLs, model identifiers, JSON/dict keys.
   - The terms **cross-sell** and **up-sell** — keep them in English (they are the canonical project vocabulary). You may write "cross-sell preporuke" / "up-sell modela" etc.
   - Endpoint names and HTTP verbs: `POST /cross-sell`, `POST /up-sell`, `GET /health`.
   - The term `CRISP-DM`.
   - Acronyms: `B2B`, `B2C`, `ERP`, `PO`, `PG` (when used as shorthand for Product Group; you may explain it in Croatian once on first use as "PG = Product group").

3. **Translate these terms consistently:**

   | English | Croatian |
   |---|---|
   | Business Understanding | Razumijevanje poslovanja |
   | Data Understanding | Razumijevanje podataka |
   | Data Preparation | Priprema podataka |
   | Modeling | Modeliranje |
   | Evaluation | Evaluacija |
   | Deployment | Implementacija (deployment) |
   | customer | kupac |
   | customer segmentation | segmentacija kupaca |
   | recommendation | preporuka |
   | recommender / recommendation system | sustav za preporuke |
   | feature / feature engineering | značajka / inženjerstvo značajki |
   | training period / train period | trening period |
   | label window / label period | label window (period oznake) |
   | snapshot date | datum snapshota |
   | cluster | klaster |
   | basket | košarica |
   | association rules | asocijacijska pravila |
   | item | proizvod / item |
   | product group | grupa proizvoda |
   | invoice | faktura |
   | gross margin | bruto marža |
   | revenue | prihod |
   | cost | trošak |
   | imbalance | neravnoteža (klasna neravnoteža) |
   | imputation | imputacija |
   | winsorize | winsorize (u zagradi: ograničavanje ekstremnih vrijednosti) |
   | premium tier / standard tier | premium / standard razred |
   | cold-start | cold-start |
   | hold-out (validation) | hold-out validacija |
   | leakage | leakage (curenje informacija) |
   | reproducibility | reproducibilnost |
   | latency | latencija |

4. **Preserve all numbers, dates, percentages, dollar amounts, and statistical values exactly as they appear.** Do not localize decimal separators (keep `.` as decimal, `,` as thousands separator — Croatian technical writing accepts both, but consistency with the source is more important than locale conventions here).

5. **Preserve all Markdown structure exactly** — same heading levels, same table layouts, same blockquotes, same horizontal rules, same bullet/numbered lists. Only the text content changes.

6. **Preserve all internal links** but translate the link text. For example:
   `see [README.md](../../README.md)` → `vidi [README.md](../../README.md)` (link text unchanged because it is a filename), but
   `see [Run Book](../runbooks/RUN_BOOK.md)` → `vidi [Run Book](../runbooks/RUN_BOOK.md)` (keep "Run Book" — it is the document's title).

7. **Diagrams (ASCII boxes):** keep the box characters and overall layout. Translate any English text inside the boxes; preserve the alignment by adjusting whitespace as needed.

8. **Tone:** professional, technically precise, no informal language. Mirror the formal-but-readable register of the English original.

9. **Do not add, remove, or reorder content.** Translate everything that is in the source file, in the same order.

10. **End with the same italicized footer line** translated to Croatian, with the same internal links.

### After translating

- Save with the Write tool to `docs/documentation/DOKUMENTACIJA.md`.
- Confirm the file size is comparable to `DOCUMENTATION.md` (within ±20%).
- Report a one-sentence summary of what was translated.

Begin by reading the source file. Do not ask follow-up questions — apply the rules and translate.
