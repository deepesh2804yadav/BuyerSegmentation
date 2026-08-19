from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clustering import evaluate_kmeans, label_clusters, run_clustering, select_k
from src.preprocessing import (
    build_client_investment_profile,
    clean_clients,
    clean_properties,
    encode_and_scale,
    parse_currency,
    parse_mixed_dates,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_mixed_dates():
    parsed = parse_mixed_dates(pd.Series(["05-11-1968", "11/26/1962", "2/28/1976"]))
    assert parsed.notna().all()
    assert parsed.dt.year.tolist() == [1968, 1962, 1976]


def test_parse_currency():
    values = parse_currency(pd.Series(["$300,385.62", "$208,930.81"]))
    assert values.tolist() == pytest.approx([300385.62, 208930.81])


def test_clean_clients_computes_age_and_normalizes():
    clients = pd.read_csv(FIXTURES / "clients_sample.csv")
    cleaned = clean_clients(clients)
    assert cleaned["client_id"].is_unique
    assert cleaned["age"].between(18, 100).all()
    assert set(cleaned["client_type"]) <= {"Individual", "Company"}
    assert cleaned.loc[cleaned["country"] != "USA", "is_international"].eq(1).all()


def test_investment_profile_aggregates_sold_units():
    clients = clean_clients(pd.read_csv(FIXTURES / "clients_sample.csv"))
    properties = clean_properties(pd.read_csv(FIXTURES / "properties_sample.csv"))
    profile = build_client_investment_profile(clients, properties)
    row = profile.set_index("client_id").loc["C0001"]
    assert row["units_purchased"] == 2
    assert row["total_investment"] == pytest.approx(500000)
    unsold_buyer = profile.set_index("client_id").loc["C0004"]
    assert unsold_buyer["units_purchased"] == 0


def test_clustering_assigns_four_named_segments():
    rng = np.random.default_rng(0)
    n = 80
    profile = pd.DataFrame(
        {
            "client_id": [f"C{i:04d}" for i in range(n)],
            "client_type": ["Company"] * 20 + ["Individual"] * 60,
            "gender": rng.choice(["M", "F"], n),
            "country": ["USA"] * 40 + ["Germany"] * 20 + ["Canada"] * 20,
            "region": rng.choice(["California", "Berlin", "Ontario"], n),
            "acquisition_purpose": ["Investment"] * 30 + ["Home"] * 50,
            "loan_applied": ["Yes"] * 25 + ["No"] * 55,
            "referral_channel": rng.choice(["Website", "Agency", "Client"], n),
            "satisfaction_score": rng.integers(1, 6, n),
            "age": rng.uniform(28, 75, n),
            "units_purchased": np.concatenate([rng.integers(4, 10, 20), rng.integers(1, 4, 60)]),
            "total_investment": np.concatenate(
                [rng.uniform(1_200_000, 2_000_000, 20), rng.uniform(150_000, 600_000, 60)]
            ),
            "avg_unit_price": rng.uniform(180_000, 500_000, n),
            "avg_floor_area_sqft": rng.uniform(600, 1600, n),
            "office_share": rng.uniform(0, 0.4, n),
            "is_international": [0] * 40 + [1] * 40,
        }
    )
    X, _, _ = encode_and_scale(profile)
    result = run_clustering(profile, X, preferred_k=4)
    assert result["k"] == 4
    assert set(result["profile"]["segment_code"]) == {"C1", "C2", "C3", "C4"}
    assert result["profile"]["buyer_type"].notna().all()
    assert 0.0 <= result["ari_kmeans_hierarchical"] <= 1.0
    companies = result["profile"].loc[result["profile"]["client_type"] == "Company"]
    assert (companies["segment_code"] == "C3").all()


def test_select_k_prefers_playbook_when_close():
    metrics = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "inertia": [10, 7, 5, 4.5],
            "silhouette": [0.40, 0.42, 0.41, 0.43],
        }
    )
    assert select_k(metrics, preferred_k=4) == 4
    metrics.loc[metrics["k"] == 5, "silhouette"] = 0.55
    assert select_k(metrics, preferred_k=4) == 5


def test_label_clusters_maps_corporate_to_c3():
    summary = pd.DataFrame(
        {
            "kmeans_cluster": [0, 1, 2, 3],
            "company_share": [0.8, 0.02, 0.01, 0.05],
            "mean_units": [6.0, 1.2, 1.1, 2.0],
            "loan_share": [0.1, 0.8, 0.2, 0.2],
            "mean_age": [50, 32, 55, 48],
            "investment_share": [0.6, 0.1, 0.7, 0.4],
            "mean_total_investment": [400_000, 200_000, 1_500_000, 300_000],
            "mean_satisfaction": [3.0, 3.1, 4.8, 3.2],
            "mean_avg_unit_price": [250_000, 180_000, 600_000, 220_000],
            "international_share": [0.2, 0.1, 0.3, 0.9],
        }
    )
    mapping = label_clusters(summary)
    assert mapping[0] == "C3"
    assert mapping[1] == "C2"
    assert mapping[2] == "C4"
    assert mapping[3] == "C1"


def test_evaluate_kmeans_returns_requested_range():
    X = np.random.default_rng(1).normal(size=(60, 4))
    metrics = evaluate_kmeans(X, k_range=range(2, 5), random_state=1)
    assert list(metrics["k"]) == [2, 3, 4]
    assert (metrics["silhouette"] > 0).all()
