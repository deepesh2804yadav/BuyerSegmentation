"""Reusable dashboard data helpers."""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
import streamlit as st

from src.paths import ARTIFACTS, DATA_PROCESSED
from src.segments import SEGMENT_PLAYBOOK

COUNTRY_ISO3 = {
    "USA": "USA",
    "UK": "GBR",
    "Canada": "CAN",
    "Germany": "DEU",
    "France": "FRA",
    "Belgium": "BEL",
    "Mexico": "MEX",
    "Australia": "AUS",
    "Russia": "RUS",
    "Denmark": "DNK",
}


@st.cache_data
def load_segments() -> pd.DataFrame:
    path = DATA_PROCESSED / "client_segments.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.train` before launching the dashboard.")
    df = pd.read_csv(path)
    df["iso3"] = df["country"].map(COUNTRY_ISO3)
    return df


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "cluster_summary.csv")


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads((ARTIFACTS / "model_metadata.json").read_text())


def apply_filters(
    df: pd.DataFrame,
    countries: list[str],
    regions: list[str],
    purposes: list[str],
    client_types: list[str],
) -> pd.DataFrame:
    out = df
    if countries:
        out = out[out["country"].isin(countries)]
    if regions:
        out = out[out["region"].isin(regions)]
    if purposes:
        out = out[out["acquisition_purpose"].isin(purposes)]
    if client_types:
        out = out[out["client_type"].isin(client_types)]
    return out


def insight_copy(segment_code: str) -> dict:
    return SEGMENT_PLAYBOOK.get(segment_code, {})
