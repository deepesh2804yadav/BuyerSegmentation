"""Data cleaning, feature engineering, encoding, and scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import DATA_RAW

REFERENCE_DATE = pd.Timestamp("2024-01-01")

# Compact feature set: high-cardinality geography is kept for EDA/filters, not clustering.
CATEGORICAL_FEATURES = ["referral_channel"]
NUMERIC_FEATURES = [
    "age",
    "satisfaction_score",
    "units_purchased",
    "log_total_investment",
    "log_avg_unit_price",
    "office_share",
]
BINARY_FEATURES = [
    "is_company",
    "is_investment",
    "is_loan",
    "is_international",
]
FEATURE_WEIGHTS = {
    "num__log_total_investment": 1.4,
    "num__units_purchased": 1.3,
    "bin__is_company": 2.4,
    "bin__is_investment": 1.5,
    "bin__is_loan": 1.4,
    "bin__is_international": 1.6,
}


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse DOB strings that mix MM-DD-YYYY and M/D/YYYY."""
    return pd.to_datetime(series, format="mixed", errors="coerce")


def parse_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan})


def load_raw_tables(
    clients_path=None,
    properties_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clients_path = clients_path or DATA_RAW / "clients.csv"
    properties_path = properties_path or DATA_RAW / "properties.csv"
    clients = pd.read_csv(clients_path)
    properties = pd.read_csv(properties_path)
    return clients, properties


def clean_clients(clients: pd.DataFrame) -> pd.DataFrame:
    df = clients.copy()
    df = df.drop_duplicates(subset=["client_id"], keep="first")

    for col in [
        "client_type",
        "gender",
        "country",
        "region",
        "acquisition_purpose",
        "loan_applied",
        "referral_channel",
    ]:
        df[col] = normalize_labels(df[col])

    df["client_type"] = df["client_type"].replace({"Corporate": "Company"})
    df["acquisition_purpose"] = df["acquisition_purpose"].replace(
        {"Personal Use": "Home", "Personal": "Home"}
    )
    df["gender"] = df["gender"].replace({"Male": "M", "Female": "F"})
    df["loan_applied"] = df["loan_applied"].replace({"True": "Yes", "False": "No", "1": "Yes", "0": "No"})

    df["date_of_birth"] = parse_mixed_dates(df["date_of_birth"])
    df["age"] = ((REFERENCE_DATE - df["date_of_birth"]).dt.days / 365.25).round(1)
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median())

    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    df["satisfaction_score"] = df["satisfaction_score"].fillna(df["satisfaction_score"].median())

    df["is_international"] = (df["country"] != "Usa").astype(int)
    df["country"] = df["country"].replace({"Usa": "USA", "Uk": "UK"})
    return df


def clean_properties(properties: pd.DataFrame) -> pd.DataFrame:
    df = properties.copy()
    df["sale_price"] = parse_currency(df["sale_price"])
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"], errors="coerce")
    df["unit_category"] = normalize_labels(df["unit_category"])
    df["listing_status"] = normalize_labels(df["listing_status"])
    df["transaction_date"] = parse_mixed_dates(df["transaction_date"])
    return df


def build_client_investment_profile(
    clients: pd.DataFrame, properties: pd.DataFrame
) -> pd.DataFrame:
    sold = properties[
        (properties["listing_status"] == "Sold") & properties["client_ref"].notna()
    ].copy()
    sold["is_office"] = (sold["unit_category"] == "Office").astype(int)

    agg = sold.groupby("client_ref").agg(
        units_purchased=("listing_id", "count"),
        total_investment=("sale_price", "sum"),
        avg_unit_price=("sale_price", "mean"),
        avg_floor_area_sqft=("floor_area_sqft", "mean"),
        office_share=("is_office", "mean"),
    )
    profile = clients.merge(agg, left_on="client_id", right_index=True, how="left")
    for col in [
        "units_purchased",
        "total_investment",
        "avg_unit_price",
        "avg_floor_area_sqft",
        "office_share",
    ]:
        profile[col] = profile[col].fillna(0)
    return add_model_features(profile)


def add_model_features(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    out["is_company"] = (out["client_type"] == "Company").astype(int)
    out["is_investment"] = (out["acquisition_purpose"] == "Investment").astype(int)
    out["is_loan"] = (out["loan_applied"] == "Yes").astype(int)
    out["log_total_investment"] = np.log1p(out["total_investment"])
    out["log_avg_unit_price"] = np.log1p(out["avg_unit_price"])
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def clustering_frame(profile: pd.DataFrame) -> pd.DataFrame:
    if "log_total_investment" not in profile.columns:
        profile = add_model_features(profile)
    cols = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    return profile[cols].copy()


def apply_feature_weights(X: np.ndarray, names: list[str]) -> np.ndarray:
    weighted = np.array(X, dtype=float, copy=True)
    for idx, name in enumerate(names):
        weighted[:, idx] *= FEATURE_WEIGHTS.get(name, 1.0)
    return weighted


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def encode_and_scale(profile: pd.DataFrame) -> tuple[np.ndarray, ColumnTransformer, list[str]]:
    X_frame = clustering_frame(profile)
    preprocessor = build_preprocessor()
    names = None
    X = preprocessor.fit_transform(X_frame)
    names = feature_names(preprocessor)
    X = apply_feature_weights(X, names)
    return X, preprocessor, names


def transform_with_fitted(
    profile: pd.DataFrame, preprocessor: ColumnTransformer
) -> np.ndarray:
    names = feature_names(preprocessor)
    X = preprocessor.transform(clustering_frame(profile))
    return apply_feature_weights(X, names)


def pipeline_artifacts_from_raw(clients_path=None, properties_path=None) -> dict[str, Any]:
    clients_raw, properties_raw = load_raw_tables(clients_path, properties_path)
    clients = clean_clients(clients_raw)
    properties = clean_properties(properties_raw)
    profile = build_client_investment_profile(clients, properties)
    X, preprocessor, names = encode_and_scale(profile)
    return {
        "clients_raw": clients_raw,
        "properties": properties,
        "profile": profile,
        "X": X,
        "preprocessor": preprocessor,
        "feature_names": names,
    }
