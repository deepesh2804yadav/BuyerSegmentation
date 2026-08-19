"""Build HTML/PDF research paper and a static dashboard page for submission links."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import plotly.express as px

from src.paths import DATA_PROCESSED, FIGURES, ROOT

DOCS = ROOT / "docs"
PALETTE = px.colors.qualitative.Set2


def b64_img(path: Path) -> str:
    mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def md_to_html(md: str) -> str:
    try:
        import markdown

        return markdown.markdown(md, extensions=["tables", "fenced_code"])
    except ImportError:
        # Minimal fallback: keep markdown readable in a <pre> if the library is missing.
        escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{escaped}</pre>"


def write_research_html() -> Path:
    md = (DOCS / "research_paper.md").read_text()
    body = md_to_html(md)
    figures = ""
    for name, caption in [
        ("elbow_method.png", "Elbow method"),
        ("silhouette_scores.png", "Silhouette scores"),
        ("pca_segments.png", "PCA of clustering features"),
        ("hierarchical_dendrogram.png", "Hierarchical dendrogram"),
    ]:
        path = FIGURES / name
        if path.exists():
            figures += (
                f'<figure><img src="{b64_img(path)}" alt="{caption}">'
                f"<figcaption>{caption}</figcaption></figure>"
            )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Buyer Segmentation Research Paper</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 880px; margin: 32px auto; padding: 0 20px;
           color: #1a1a1a; line-height: 1.55; }}
    h1, h2, h3 {{ font-family: "Helvetica Neue", Arial, sans-serif; color: #123; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef4f6; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 100%; }}
    figcaption {{ font-size: 13px; color: #555; text-align: center; }}
  </style>
</head>
<body>
{body}
<h2>Appendix: model diagnostics</h2>
{figures}
</body>
</html>
"""
    out = DOCS / "research_paper.html"
    out.write_text(html)
    return out


def write_dashboard_html() -> Path:
    df = pd.read_csv(DATA_PROCESSED / "client_segments.csv")
    summary = pd.read_csv(DATA_PROCESSED / "cluster_summary.csv")
    dist = df.groupby(["segment_code", "buyer_type"], as_index=False).size()
    dist["label"] = dist["segment_code"] + " · " + dist["buyer_type"]
    pie = px.pie(dist, names="label", values="size", color_discrete_sequence=PALETTE, hole=0.45,
                 title="Cluster distribution")
    spend = df.groupby("segment_label", as_index=False)["total_investment"].sum()
    bar = px.bar(spend, x="segment_label", y="total_investment", color="segment_label",
                 color_discrete_sequence=PALETTE, title="Aggregate investment by segment")
    bar.update_layout(showlegend=False)
    purpose = df.groupby(["segment_label", "acquisition_purpose"]).size().reset_index(name="buyers")
    purpose_bar = px.bar(purpose, x="segment_label", y="buyers", color="acquisition_purpose",
                         barmode="stack", title="Acquisition purpose by segment",
                         color_discrete_sequence=PALETTE)
    country = df.groupby("country", as_index=False).size().rename(columns={"size": "buyers"})
    geo = px.bar(country.sort_values("buyers", ascending=False), x="country", y="buyers",
                 title="Buyers by country", color_discrete_sequence=PALETTE)
    table_html = summary.to_html(index=False)
    parts = [
        pie.to_html(full_html=False, include_plotlyjs="cdn"),
        bar.to_html(full_html=False, include_plotlyjs=False),
        purpose_bar.to_html(full_html=False, include_plotlyjs=False),
        geo.to_html(full_html=False, include_plotlyjs=False),
    ]
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Parcl Buyer Intelligence Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7fafb; color: #123; }}
    h1 {{ margin-bottom: 8px; }}
    .note {{ color: #456; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #e8f1f3; }}
  </style>
</head>
<body>
  <h1>Parcl Buyer Segmentation &amp; Investment Profiling</h1>
  <p class="note">Static web view of the trained segments (2,000 buyers). For live filters use
  <code>streamlit run app.py</code> or the Streamlit Cloud deployment.</p>
  {''.join(parts)}
  <h2>Cluster summary</h2>
  {table_html}
</body>
</html>
"""
    out = DOCS / "dashboard.html"
    out.write_text(html)
    return out


def write_pdf(html_path: Path) -> Path:
    """Write a simple text PDF from the markdown source (no browser dependency)."""
    from fpdf import FPDF

    md = (DOCS / "research_paper.md").read_text()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for raw_line in md.splitlines():
        line = raw_line.encode("latin-1", "replace").decode("latin-1")
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, line[3:])
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("|") or line.startswith("- ") or line.strip():
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, line.replace("**", ""))
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.ln(2)
    out = DOCS / "research_paper.pdf"
    pdf.output(str(out))
    return out


if __name__ == "__main__":
    try:
        import markdown  # noqa: F401
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "-q"])
    html = write_research_html()
    dash = write_dashboard_html()
    pdf = write_pdf(html)
    print(html)
    print(dash)
    print(pdf, pdf.stat().st_size)
