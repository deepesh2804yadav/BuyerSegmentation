"""K-Means and hierarchical clustering, evaluation, and segment labeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEGMENT_PLAYBOOK = {
    "C1": {
        "buyer_type": "Global Investors",
        "characteristics": "Highest share of non-US buyers and solid investment-purpose demand.",
        "marketing": "Target with yield, FX-aware pricing, and cross-border investment packs.",
    },
    "C2": {
        "buyer_type": "First-Time Buyers",
        "characteristics": "Largest owner-occupier pool: home-led purchases with frequent loan use.",
        "marketing": "Lead with financing partners, starter-to-mid inventory, and education content.",
    },
    "C3": {
        "buyer_type": "Corporate Buyers",
        "characteristics": "Registered company accounts. Younger decision-makers buying multiple units.",
        "marketing": "Offer bulk pricing, office mix, and relationship-managed deals.",
    },
    "C4": {
        "buyer_type": "Luxury Investors",
        "characteristics": "Small high-value cohort: older buyers, higher satisfaction, largest portfolios.",
        "marketing": "Prioritize concierge sales, premium towers, and exclusive listings.",
    },
}


def evaluate_kmeans(
    X: np.ndarray, k_range: range = range(2, 9), random_state: int = 42
) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
            }
        )
    return pd.DataFrame(rows)


def select_k(metrics: pd.DataFrame, preferred_k: int = 4, tolerance: float = 0.03) -> int:
    """Prefer the playbook k when silhouette is competitive; otherwise max silhouette."""
    best_k = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    preferred = metrics.loc[metrics["k"] == preferred_k]
    if preferred.empty:
        return best_k
    preferred_sil = float(preferred["silhouette"].iloc[0])
    best_sil = float(metrics["silhouette"].max())
    if preferred_sil >= best_sil - tolerance:
        return preferred_k
    return best_k


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    model.fit(X)
    return model


def fit_hierarchical(X: np.ndarray, k: int) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(X)


def cluster_profile_table(profile: pd.DataFrame, cluster_col: str = "kmeans_cluster") -> pd.DataFrame:
    grouped = profile.groupby(cluster_col)
    summary = grouped.agg(
        buyers=("client_id", "count"),
        company_share=("client_type", lambda s: (s == "Company").mean()),
        investment_share=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan_share=("loan_applied", lambda s: (s == "Yes").mean()),
        international_share=("is_international", "mean"),
        mean_age=("age", "mean"),
        mean_satisfaction=("satisfaction_score", "mean"),
        mean_units=("units_purchased", "mean"),
        mean_total_investment=("total_investment", "mean"),
        median_total_investment=("total_investment", "median"),
        mean_avg_unit_price=("avg_unit_price", "mean"),
    )
    summary["pct_of_base"] = summary["buyers"] / summary["buyers"].sum()
    return summary.reset_index()


def _assign_unique(scores: dict[int, float], taken: set[int]) -> int | None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for cluster_id, _ in ranked:
        if cluster_id not in taken:
            return int(cluster_id)
    return None


def _map_codes(summary: pd.DataFrame, score_specs: list[tuple[str, pd.Series]]) -> dict[int, str]:
    rows = summary.set_index("kmeans_cluster")
    cluster_ids = [int(i) for i in rows.index]
    taken: set[int] = set()
    mapping: dict[int, str] = {}
    for code, scores in score_specs:
        cluster_id = _assign_unique(scores.to_dict(), taken)
        if cluster_id is None:
            break
        mapping[cluster_id] = code
        taken.add(cluster_id)
    leftover_codes = [code for code, _ in score_specs if code not in mapping.values()]
    for cluster_id in cluster_ids:
        if cluster_id not in mapping:
            mapping[cluster_id] = leftover_codes.pop(0) if leftover_codes else f"C{cluster_id}"
    return mapping


def label_clusters(summary: pd.DataFrame) -> dict[int, str]:
    """Map numeric clusters onto the four recommended buyer types using data traits."""
    rows = summary.set_index("kmeans_cluster")
    return _map_codes(
        summary,
        [
            ("C3", rows["company_share"] + 0.15 * rows["mean_units"]),
            (
                "C2",
                rows["loan_share"]
                + (1 - rows["mean_age"] / rows["mean_age"].max())
                + (1 - rows["investment_share"])
                - rows["company_share"],
            ),
            (
                "C4",
                rows["mean_total_investment"] / rows["mean_total_investment"].max()
                + rows["mean_satisfaction"] / rows["mean_satisfaction"].max()
                + rows["mean_avg_unit_price"] / rows["mean_avg_unit_price"].max(),
            ),
            ("C1", rows["international_share"] + rows["investment_share"]),
        ],
    )


def label_individual_clusters(summary: pd.DataFrame) -> dict[int, str]:
    """Name the non-corporate K-Means groups as C1/C2/C4."""
    rows = summary.set_index("kmeans_cluster")
    return _map_codes(
        summary,
        [
            (
                "C2",
                rows["loan_share"]
                + (1 - rows["mean_age"] / rows["mean_age"].max())
                + (1 - rows["investment_share"]),
            ),
            (
                "C4",
                rows["mean_total_investment"] / rows["mean_total_investment"].max()
                + rows["mean_satisfaction"] / rows["mean_satisfaction"].max()
                + rows["mean_avg_unit_price"] / rows["mean_avg_unit_price"].max(),
            ),
            ("C1", rows["international_share"] + rows["investment_share"]),
        ],
    )


def attach_segment_metadata(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    meta = out["segment_code"].map(SEGMENT_PLAYBOOK)
    out["buyer_type"] = meta.map(lambda d: d["buyer_type"] if isinstance(d, dict) else "Unlabeled")
    out["segment_label"] = out["segment_code"] + " · " + out["buyer_type"]
    return out


def run_clustering(profile: pd.DataFrame, X: np.ndarray, preferred_k: int = 4) -> dict[str, Any]:
    """Hybrid segmentation: companies are C3; remaining buyers are clustered into C1/C2/C4."""
    metrics = evaluate_kmeans(X)
    unconstrained_k = select_k(metrics, preferred_k=preferred_k, tolerance=0.15)

    labeled = profile.reset_index(drop=True).copy()
    X = np.asarray(X)
    company_mask = labeled["client_type"].eq("Company").to_numpy()
    individual_idx = np.flatnonzero(~company_mask)
    k_ind = max(preferred_k - 1, 2)
    company_cluster_id = k_ind

    labeled["kmeans_cluster"] = company_cluster_id
    labeled["hierarchical_cluster"] = company_cluster_id
    kmeans = None
    if len(individual_idx) >= k_ind:
        kmeans = fit_kmeans(X[individual_idx], k_ind)
        hier_ind = fit_hierarchical(X[individual_idx], k_ind)
        labeled.loc[~company_mask, "kmeans_cluster"] = kmeans.labels_
        labeled.loc[~company_mask, "hierarchical_cluster"] = hier_ind
        code_map = label_individual_clusters(
            cluster_profile_table(labeled.loc[~company_mask])
        )
    else:
        kmeans = fit_kmeans(X, preferred_k)
        labeled["kmeans_cluster"] = kmeans.labels_
        labeled["hierarchical_cluster"] = fit_hierarchical(X, preferred_k)
        code_map = label_clusters(cluster_profile_table(labeled))

    code_map[int(company_cluster_id)] = "C3"
    labeled["kmeans_cluster"] = labeled["kmeans_cluster"].astype(int)
    labeled["hierarchical_cluster"] = labeled["hierarchical_cluster"].astype(int)
    labeled["segment_code"] = labeled["kmeans_cluster"].map(code_map)
    labeled.loc[company_mask, "segment_code"] = "C3"
    labeled = attach_segment_metadata(labeled)

    summary = cluster_profile_table(labeled)
    summary["segment_code"] = summary["kmeans_cluster"].map(
        lambda cid: "C3" if int(cid) == int(company_cluster_id) else code_map.get(int(cid))
    )
    summary["buyer_type"] = summary["segment_code"].map(
        lambda c: SEGMENT_PLAYBOOK.get(c, {}).get("buyer_type", "Unlabeled")
    )

    agreement = float(
        adjusted_rand_score(labeled["kmeans_cluster"], labeled["hierarchical_cluster"])
    )
    kmeans_sil = float(silhouette_score(X, labeled["kmeans_cluster"]))
    hier_sil = float(silhouette_score(X, labeled["hierarchical_cluster"]))

    return {
        "k": preferred_k,
        "unconstrained_k": unconstrained_k,
        "kmeans": kmeans,
        "metrics": metrics,
        "profile": labeled,
        "summary": summary.sort_values("segment_code"),
        "code_map": {int(k): v for k, v in code_map.items()},
        "company_cluster_id": int(company_cluster_id),
        "ari_kmeans_hierarchical": agreement,
        "kmeans_silhouette": kmeans_sil,
        "hierarchical_silhouette": hier_sil,
    }
