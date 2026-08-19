"""Streamlit buyer segmentation and investment profiling dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_data import apply_filters, insight_copy, load_metadata, load_segments, load_summary
from src.paths import FIGURES

st.set_page_config(
    page_title="Parcl Buyer Intelligence",
    page_icon="🏢",
    layout="wide",
)

PALETTE = px.colors.qualitative.Set2


def kpi(label: str, value, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def money(v: float) -> str:
    if pd.isna(v):
        return "—"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"


def pct(v: float) -> str:
    return f"{v*100:.1f}%"


def main() -> None:
    try:
        clients = load_segments()
        summary = load_summary()
        meta = load_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.title("Parcl Buyer Segmentation & Investment Profiling")
    st.caption(
        "Machine-learning clustering of buyer behavior for Unified Mentor × Parcl market intelligence."
    )

    with st.sidebar:
        st.header("Filters")
        countries = st.multiselect("Country", sorted(clients["country"].unique()))
        regions = st.multiselect("Region", sorted(clients["region"].unique()))
        purposes = st.multiselect(
            "Acquisition purpose", sorted(clients["acquisition_purpose"].unique())
        )
        client_types = st.multiselect("Client type", sorted(clients["client_type"].unique()))
        st.divider()
        st.markdown(
            f"**Model:** hybrid k={meta['k']}  \n"
            f"**Silhouette:** {meta['kmeans_silhouette']:.3f}  \n"
            f"**K-Means vs Hierarchical ARI:** {meta['ari_kmeans_hierarchical']:.3f}  \n"
            f"**Unconstrained K-Means k:** {meta.get('unconstrained_k', '—')}"
        )
        st.caption(
            "Company accounts are tagged as C3. Individual buyers are clustered with K-Means "
            "and validated with hierarchical clustering."
        )

    filtered = apply_filters(clients, countries, regions, purposes, client_types)
    if filtered.empty:
        st.warning("No buyers match the current filters.")
        st.stop()

    overview, behavior, geo, insights = st.tabs(
        [
            "Buyer Segmentation Overview",
            "Investor Behavior Dashboard",
            "Geographic Buyer Analysis",
            "Segment Insights Panel",
        ]
    )

    with overview:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Buyers in view", f"{len(filtered):,}")
        with c2:
            kpi("Total invested", money(filtered["total_investment"].sum()))
        with c3:
            kpi("Avg satisfaction", f"{filtered['satisfaction_score'].mean():.2f}")
        with c4:
            kpi("Loan application rate", pct((filtered["loan_applied"] == "Yes").mean()))

        dist = (
            filtered.groupby(["segment_code", "buyer_type"], as_index=False)
            .size()
            .rename(columns={"size": "buyers"})
        )
        dist["label"] = dist["segment_code"] + " · " + dist["buyer_type"]
        fig = px.pie(
            dist,
            names="label",
            values="buyers",
            color="label",
            color_discrete_sequence=PALETTE,
            hole=0.45,
            title="Cluster distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            purpose = (
                filtered.groupby(["segment_label", "acquisition_purpose"])
                .size()
                .reset_index(name="buyers")
            )
            fig = px.bar(
                purpose,
                x="segment_label",
                y="buyers",
                color="acquisition_purpose",
                barmode="stack",
                title="Acquisition purpose by segment",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            ctype = (
                filtered.groupby(["segment_label", "client_type"])
                .size()
                .reset_index(name="buyers")
            )
            fig = px.bar(
                ctype,
                x="segment_label",
                y="buyers",
                color="client_type",
                barmode="stack",
                title="Client type mix by segment",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

        elbow = FIGURES / "elbow_method.png"
        silhouette = FIGURES / "silhouette_scores.png"
        if elbow.exists() and silhouette.exists():
            st.subheader("Model diagnostics")
            img1, img2 = st.columns(2)
            img1.image(str(elbow), caption="Elbow method")
            img2.image(str(silhouette), caption="Silhouette scores")
            pca_path = FIGURES / "pca_segments.png"
            dendro_path = FIGURES / "hierarchical_dendrogram.png"
            extra1, extra2 = st.columns(2)
            if pca_path.exists():
                extra1.image(str(pca_path), caption="PCA projection of buyer features")
            if dendro_path.exists():
                extra2.image(str(dendro_path), caption="Hierarchical structure (Ward linkage)")

    with behavior:
        st.subheader("Investment patterns by cluster")
        spend = filtered.groupby("segment_label", as_index=False).agg(
            total_investment=("total_investment", "sum"),
            avg_ticket=("avg_unit_price", "mean"),
            avg_units=("units_purchased", "mean"),
            loan_rate=("loan_applied", lambda s: (s == "Yes").mean()),
            office_share=("office_share", "mean"),
        )
        fig = px.bar(
            spend,
            x="segment_label",
            y="total_investment",
            color="segment_label",
            title="Aggregate investment by segment",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        m1, m2 = st.columns(2)
        with m1:
            fig = px.box(
                filtered,
                x="segment_label",
                y="total_investment",
                color="segment_label",
                title="Distribution of buyer spend",
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with m2:
            fig = px.scatter(
                filtered,
                x="age",
                y="avg_unit_price",
                color="segment_label",
                size="units_purchased",
                hover_data=["client_id", "country", "acquisition_purpose"],
                title="Age vs average unit price",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

        loan = (
            filtered.groupby(["segment_label", "loan_applied"])
            .size()
            .reset_index(name="buyers")
        )
        channel = (
            filtered.groupby(["segment_label", "referral_channel"])
            .size()
            .reset_index(name="buyers")
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                loan,
                x="segment_label",
                y="buyers",
                color="loan_applied",
                barmode="group",
                title="Financing behavior",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                channel,
                x="segment_label",
                y="buyers",
                color="referral_channel",
                barmode="stack",
                title="Referral channel mix",
                color_discrete_sequence=PALETTE,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            spend.assign(
                total_investment=lambda d: d["total_investment"].map(money),
                avg_ticket=lambda d: d["avg_ticket"].map(money),
                avg_units=lambda d: d["avg_units"].map(lambda x: f"{x:.2f}"),
                loan_rate=lambda d: d["loan_rate"].map(pct),
                office_share=lambda d: d["office_share"].map(pct),
            ),
            use_container_width=True,
            hide_index=True,
        )

    with geo:
        st.subheader("Buyer segments by geography")
        country_counts = (
            filtered.dropna(subset=["iso3"])
            .groupby(["iso3", "country"], as_index=False)
            .size()
            .rename(columns={"size": "buyers"})
        )
        fig = px.choropleth(
            country_counts,
            locations="iso3",
            color="buyers",
            hover_name="country",
            color_continuous_scale="Teal",
            title="Buyers by country of residence",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        region_seg = (
            filtered.groupby(["region", "segment_label"])
            .size()
            .reset_index(name="buyers")
            .sort_values("buyers", ascending=False)
        )
        top_regions = region_seg.groupby("region")["buyers"].sum().nlargest(15).index
        fig = px.bar(
            region_seg[region_seg["region"].isin(top_regions)],
            x="region",
            y="buyers",
            color="segment_label",
            title="Top 15 regions by buyer volume",
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

        intl = filtered.groupby(["country", "segment_label"], as_index=False).agg(
            buyers=("client_id", "count"),
            investment=("total_investment", "sum"),
        )
        fig = px.treemap(
            intl,
            path=["country", "segment_label"],
            values="buyers",
            color="investment",
            color_continuous_scale="Teal",
            title="Country × segment composition (sized by buyers, colored by spend)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with insights:
        st.subheader("Descriptive statistics per cluster")
        view_summary = summary.copy()
        display = pd.DataFrame(
            {
                "Cluster": view_summary["segment_code"],
                "Buyer type": view_summary["buyer_type"],
                "Buyers": view_summary["buyers"],
                "Share of base": view_summary["pct_of_base"].map(pct),
                "Company share": view_summary["company_share"].map(pct),
                "Investment purpose": view_summary["investment_share"].map(pct),
                "Loan share": view_summary["loan_share"].map(pct),
                "International": view_summary["international_share"].map(pct),
                "Mean age": view_summary["mean_age"].round(1),
                "Satisfaction": view_summary["mean_satisfaction"].round(2),
                "Mean units": view_summary["mean_units"].round(2),
                "Mean spend": view_summary["mean_total_investment"].map(money),
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        selected = st.selectbox(
            "Inspect a segment",
            sorted(filtered["segment_code"].unique()),
            format_func=lambda code: f"{code} · {insight_copy(code).get('buyer_type', code)}",
        )
        copy = insight_copy(selected)
        subset = filtered[filtered["segment_code"] == selected]
        st.markdown(f"### {selected} · {copy.get('buyer_type', '')}")
        st.write(copy.get("characteristics", ""))
        st.info(f"Recommended action: {copy.get('marketing', '')}")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Buyers", f"{len(subset):,}")
        k2.metric("Mean age", f"{subset['age'].mean():.1f}")
        k3.metric("Mean spend", money(subset["total_investment"].mean()))
        k4.metric("Loan rate", pct((subset["loan_applied"] == "Yes").mean()))
        k5.metric("Investment purpose", pct((subset["acquisition_purpose"] == "Investment").mean()))

        top_countries = subset["country"].value_counts().head(8).rename_axis("country").reset_index(name="buyers")
        fig = px.bar(
            top_countries,
            x="country",
            y="buyers",
            title=f"{selected} geographic concentration",
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Sample buyers")
        st.dataframe(
            subset[
                [
                    "client_id",
                    "client_type",
                    "country",
                    "region",
                    "age",
                    "acquisition_purpose",
                    "loan_applied",
                    "referral_channel",
                    "satisfaction_score",
                    "units_purchased",
                    "total_investment",
                ]
            ]
            .sort_values("total_investment", ascending=False)
            .head(25),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
