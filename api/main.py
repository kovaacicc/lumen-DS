from contextlib import asynccontextmanager
from pathlib import Path
import json
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models" / "recommender"
DATA_REC = BASE / "data" / "processed" / "recommender"
DATA_UP = BASE / "data" / "processed" / "upsell"
DATA_PROC = BASE / "data" / "processed"

TOP_K = 5

FEATURE_COLS = [
    "train_revenue", "train_orders", "train_items", "train_avg_gm",
    "train_n_groups", "recency_days", "tenure_days", "order_frequency",
    "region_enc", "industry_enc", "top_group_enc", "cluster_enc",
    "R_score", "F_score", "M_score",
    "pg_n_customers", "pg_avg_price", "pg_avg_gm",
    "pg_total_orders", "pg_avg_revenue", "pg_popularity", "pg_enc",
    "rule_exists", "rule_max_lift", "rule_max_confidence",
    "max_copurchase_rate", "mean_copurchase_rate",
]

CUST_COLS = [
    "train_revenue", "train_orders", "train_items", "train_avg_gm",
    "train_n_groups", "recency_days", "tenure_days", "order_frequency",
    "region_enc", "industry_enc", "top_group_enc", "cluster_enc",
    "R_score", "F_score", "M_score",
]

state: dict = {}


# ── Startup: load everything once ─────────────────────────────────────────────

def _load_data():
    state["xgb"] = joblib.load(MODELS_DIR / "recommender_xgb.pkl")

    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    state["all_product_groups"] = meta["all_product_groups"]

    # Customer features (known customers only)
    cf_path = DATA_REC / "customer_feats.csv"
    if cf_path.exists():
        cf = pd.read_csv(cf_path)
        state["customer_feats"] = cf.set_index("CustomerID").to_dict(orient="index")
    else:
        state["customer_feats"] = {}

    # Which PGs each customer already bought
    cpp_path = DATA_REC / "customer_purchased_pgs.csv"
    if cpp_path.exists():
        cpp = pd.read_csv(cpp_path)
        state["customer_purchased_pgs"] = dict(
            zip(cpp["CustomerID"], cpp["purchased_pgs"].fillna(""))
        )
    else:
        state["customer_purchased_pgs"] = {}

    # Product group features
    pf_path = DATA_REC / "product_feats.csv"
    if pf_path.exists():
        pf = pd.read_csv(pf_path)
        state["product_feats"] = pf.set_index("product_group").to_dict(orient="index")
    else:
        state["product_feats"] = {}

    # Median feature values — default for new/cold-start customers
    fm_path = DATA_REC / "feature_medians.json"
    if fm_path.exists():
        with open(fm_path) as f:
            state["feature_medians"] = json.load(f)
    else:
        state["feature_medians"] = {col: 0.0 for col in CUST_COLS}

    # Directed co-purchase rates: (existing_pg, candidate_pg) → rate
    cr_path = DATA_REC / "copurchase_rates.csv"
    if cr_path.exists():
        cr = pd.read_csv(cr_path)
        state["copurchase_rates"] = {
            (row["existing_pg"], row["product_group"]): row["copurchase_rate"]
            for _, row in cr.iterrows()
        }
    else:
        state["copurchase_rates"] = {}

    # Association rules lookup: (frozenset antecedents, consequent) → {lift, confidence}
    rules_df = pd.read_csv(DATA_PROC / "association_rules.csv")
    rules_lookup: dict = {}
    for _, row in rules_df.iterrows():
        ants = frozenset(row["antecedents_str"].split(", "))
        con = row["consequents_str"].strip()
        key = (ants, con)
        if key not in rules_lookup or row["lift"] > rules_lookup[key]["lift"]:
            rules_lookup[key] = {"lift": float(row["lift"]), "confidence": float(row["confidence"])}
    state["rules_lookup"] = rules_lookup

    # Up-sell: item tiers (standard vs premium per item).
    # A small number of items appear in >1 PG — keep first occurrence.
    state["item_tiers"] = (
        pd.read_csv(DATA_UP / "item_tiers.csv")
        .drop_duplicates("Item Code", keep="first")
        .set_index("Item Code")
        .to_dict(orient="index")
    )

    # Up-sell: cluster-conditional premium item popularity
    state["upsell_popularity"] = pd.read_csv(DATA_UP / "upsell_item_popularity.csv")

    # Cross-sell Layer 2: all items by cluster (no premium filter)
    all_pop_path = DATA_UP / "upsell_item_popularity_all.csv"
    state["item_popularity_all"] = (
        pd.read_csv(all_pop_path) if all_pop_path.exists() else None
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_data()
    yield


app = FastAPI(title="Lumen Recommendation API", version="1.0", lifespan=lifespan)


# ── Feature helpers ───────────────────────────────────────────────────────────

def _rule_features(selected_products: List[str], candidate_pg: str):
    selected_set = frozenset(selected_products)
    matching = [
        v for (ants, con), v in state["rules_lookup"].items()
        if ants.issubset(selected_set) and con == candidate_pg
    ]
    if not matching:
        return 0, 0.0, 0.0
    return (
        1,
        max(r["lift"] for r in matching),
        max(r["confidence"] for r in matching),
    )


def _copurchase_features(selected_products: List[str], candidate_pg: str):
    rates = [
        state["copurchase_rates"].get((sp, candidate_pg), 0.0)
        for sp in selected_products
        if sp != candidate_pg
    ]
    if not rates:
        return 0.0, 0.0
    return float(max(rates)), float(np.mean(rates))


def _items_to_pgs(item_codes: List[str]) -> List[str]:
    """Map specific item codes to their product groups using the item_tiers lookup."""
    pgs = []
    seen = set()
    for code in item_codes:
        info = state["item_tiers"].get(code)
        if info and info["Product group"] not in seen:
            pgs.append(info["Product group"])
            seen.add(info["Product group"])
    return pgs


def _layer2_items(pg: str, cluster_name: Optional[str]) -> List[dict]:
    pop = state.get("item_popularity_all")
    if pop is None:
        return []
    subset = pop[pop["Product group"] == pg]
    if cluster_name:
        cluster_sub = subset[subset["Cluster_name"] == cluster_name]
        if not cluster_sub.empty:
            subset = cluster_sub
    top = subset.nlargest(TOP_K, "confidence")
    return [
        {
            "item_code": row["Item Code"],
            "confidence": round(float(row["confidence"]), 4),
            "avg_price": round(float(row["item_avg_price"]), 2),
        }
        for _, row in top.iterrows()
    ]


# ── Request / response schemas ────────────────────────────────────────────────

class CrossSellRequest(BaseModel):
    customer_id: Optional[int] = None
    item_codes: Optional[List[str]] = None  # item codes the customer has selected / currently orders


class UpsellRequest(BaseModel):
    customer_id: Optional[int] = None
    item_codes: List[str]  # specific item codes to check for up-sell


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "xgb" in state,
        "known_customers": len(state.get("customer_feats", {})),
        "layer2_available": state.get("item_popularity_all") is not None,
    }


@app.post("/cross-sell")
def cross_sell(req: CrossSellRequest):
    customer_id = req.customer_id

    # Map incoming item codes → PG codes for co-purchase signal
    selected_pgs = _items_to_pgs(req.item_codes or [])

    # Customer-level features
    if customer_id and customer_id in state["customer_feats"]:
        cust = state["customer_feats"][customer_id]
        cluster_name = cust.get("Cluster_name")
    else:
        cust = state["feature_medians"]
        cluster_name = None

    # PGs this customer already buys (excluded from recommendations)
    if customer_id and customer_id in state["customer_purchased_pgs"]:
        raw = state["customer_purchased_pgs"][customer_id]
        purchased = set(raw.split("|")) if raw else set()
        purchased.update(selected_pgs)  # also exclude PGs from the current order
    else:
        purchased = set(selected_pgs)

    candidates = [pg for pg in state["all_product_groups"] if pg not in purchased]
    if not candidates:
        return {"customer_id": customer_id, "recommendations": []}

    # Build feature rows for every candidate PG
    product_feats = state["product_feats"]
    pg_list, feat_rows = [], []
    for pg in candidates:
        pg_f = product_feats.get(pg, {})
        rule_exists, rule_lift, rule_conf = _rule_features(selected_pgs, pg)
        max_cp, mean_cp = _copurchase_features(selected_pgs, pg)

        row = {col: cust.get(col, 0.0) for col in CUST_COLS}
        row.update(
            pg_n_customers=pg_f.get("pg_n_customers", 0.0),
            pg_avg_price=pg_f.get("pg_avg_price", 0.0),
            pg_avg_gm=pg_f.get("pg_avg_gm", 0.0),
            pg_total_orders=pg_f.get("pg_total_orders", 0.0),
            pg_avg_revenue=pg_f.get("pg_avg_revenue", 0.0),
            pg_popularity=pg_f.get("pg_popularity", 0.0),
            pg_enc=pg_f.get("pg_enc", 0.0),
            rule_exists=rule_exists,
            rule_max_lift=rule_lift,
            rule_max_confidence=rule_conf,
            max_copurchase_rate=max_cp,
            mean_copurchase_rate=mean_cp,
        )
        pg_list.append(pg)
        feat_rows.append(row)

    X = pd.DataFrame(feat_rows)[FEATURE_COLS].fillna(0.0)
    probs = state["xgb"].predict_proba(X)[:, 1]

    scores = sorted(zip(pg_list, probs.tolist()), key=lambda x: x[1], reverse=True)

    recommendations = []
    for pg, confidence in scores[:TOP_K]:
        recommendations.append(
            {
                "product_group": pg,
                "confidence": round(confidence, 4),
                "items": _layer2_items(pg, cluster_name),
            }
        )

    return {"customer_id": customer_id, "recommendations": recommendations}


@app.post("/up-sell")
def up_sell(req: UpsellRequest):
    customer_id = req.customer_id

    # Cluster for personalised item ranking
    if customer_id and customer_id in state["customer_feats"]:
        cluster_name = state["customer_feats"][customer_id].get("Cluster_name")
    else:
        cluster_name = None

    item_tiers = state["item_tiers"]
    pop = state["upsell_popularity"]

    recommendations = []
    for item_code in req.item_codes:
        if item_code not in item_tiers:
            continue
        info = item_tiers[item_code]
        tier = info["tier"]
        pg = info["Product group"]

        if tier == "premium":
            continue  # already the premium tier

        # Premium alternatives in the same product group
        subset = pop[pop["Product group"] == pg]
        if cluster_name:
            cluster_sub = subset[subset["Cluster_name"] == cluster_name]
            if not cluster_sub.empty:
                subset = cluster_sub

        top_alts = subset[subset["Item Code"] != item_code].nlargest(TOP_K, "confidence")
        if top_alts.empty:
            continue

        recommendations.append(
            {
                "current_item": item_code,
                "product_group": pg,
                "current_tier": tier,
                "current_avg_price": round(float(info["item_avg_price"]), 2),
                "premium_alternatives": [
                    {
                        "item_code": r["Item Code"],
                        "confidence": round(float(r["confidence"]), 4),
                        "avg_price": round(float(r["item_avg_price"]), 2),
                    }
                    for _, r in top_alts.iterrows()
                ],
            }
        )

    return {"customer_id": customer_id, "recommendations": recommendations}
