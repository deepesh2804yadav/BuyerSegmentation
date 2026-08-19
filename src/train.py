"""Train clustering models and persist dashboard artifacts."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from src.clustering import run_clustering
from src.paths import ARTIFACTS, DATA_PROCESSED, FIGURES, MODELS, ensure_output_dirs
from src.preprocessing import pipeline_artifacts_from_raw


def save_evaluation_plots(metrics: pd.DataFrame, k: int, X, segment_labels) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.axvline(k, color="crimson", linestyle="--", label=f"selected k={k}")
    ax.set_title("Elbow method (K-Means inertia)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "elbow_method.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(metrics["k"], metrics["silhouette"], marker="o", color="teal")
    ax.axvline(k, color="crimson", linestyle="--", label=f"selected k={k}")
    ax.set_title("Silhouette score by k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "silhouette_scores.png", dpi=140)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    scatter = pd.DataFrame(
        {"pc1": coords[:, 0], "pc2": coords[:, 1], "segment": segment_labels}
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=scatter, x="pc1", y="pc2", hue="segment", ax=ax, s=18, palette="Set2")
    ax.set_title(
        f"PCA of clustering features (explained var {pca.explained_variance_ratio_.sum():.0%})"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "pca_segments.png", dpi=140)
    plt.close(fig)

    sample_n = min(80, len(X))
    sample = X[:sample_n]
    linked = linkage(sample, method="ward")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linked, truncate_mode="level", p=5, ax=ax, no_labels=True)
    ax.set_title("Hierarchical clustering dendrogram (Ward, 80-buyer sample)")
    fig.tight_layout()
    fig.savefig(FIGURES / "hierarchical_dendrogram.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    prepared = pipeline_artifacts_from_raw()
    result = run_clustering(prepared["profile"], prepared["X"])

    profile = result["profile"]
    profile.to_csv(DATA_PROCESSED / "client_segments.csv", index=False)
    result["summary"].to_csv(DATA_PROCESSED / "cluster_summary.csv", index=False)
    result["metrics"].to_csv(DATA_PROCESSED / "kmeans_metrics.csv", index=False)

    joblib.dump(result["kmeans"], MODELS / "kmeans.joblib")
    joblib.dump(prepared["preprocessor"], MODELS / "preprocessor.joblib")

    metadata = {
        "k": result["k"],
        "unconstrained_k": result.get("unconstrained_k"),
        "company_cluster_id": result.get("company_cluster_id"),
        "kmeans_silhouette": result["kmeans_silhouette"],
        "hierarchical_silhouette": result["hierarchical_silhouette"],
        "ari_kmeans_hierarchical": result["ari_kmeans_hierarchical"],
        "code_map": {str(k): v for k, v in result["code_map"].items()},
        "feature_names": prepared["feature_names"],
        "n_clients": int(len(profile)),
        "n_sold_listings": int((prepared["properties"]["listing_status"] == "Sold").sum()),
        "segmentation": "Companies are assigned to C3; K-Means (k=3) plus hierarchical validation segment individuals into C1/C2/C4.",
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    save_evaluation_plots(
        result["metrics"],
        result["k"],
        prepared["X"],
        result["profile"]["segment_label"],
    )

    print(json.dumps({k: metadata[k] for k in ["k", "kmeans_silhouette", "ari_kmeans_hierarchical"]}, indent=2))
    print(result["summary"][["segment_code", "buyer_type", "buyers", "mean_age", "loan_share", "company_share", "mean_total_investment"]].to_string(index=False))


if __name__ == "__main__":
    main()
