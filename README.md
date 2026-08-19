# Buyer Segmentation & Investment Profiling

Machine-learning buyer intelligence for the Parcl real estate platform, prepared for the Unified Mentor × Parcl Co. Limited market-intelligence brief.

The pipeline cleans client and listing data, engineers investment profiles, clusters buyers with K-Means (validated by hierarchical clustering), and serves a Streamlit dashboard for segment targeting.

## Dataset

| File | Rows | Role |
| --- | --- | --- |
| `data/raw/clients.csv` | 2,000 | Buyer demographics, purpose, financing, satisfaction |
| `data/raw/properties.csv` | 10,000 | Unit sales used to build spend / portfolio features |

Client fields: `client_id`, `client_type`, `gender`, `country`, `region`, `date_of_birth`, `acquisition_purpose`, `loan_applied`, `referral_channel`, `satisfaction_score`.

## Methodology

1. **Cleaning** — mixed date formats, currency strings, duplicate `client_id`s, normalized categoricals, age at 2024-01-01.
2. **Encoding** — StandardScaler on numeric portfolio features; binary flags for company / investment / loan / international; one-hot `referral_channel`.
3. **Scaling / weighting** — spend, company status, and cross-border flags are up-weighted so clusters follow investment behavior rather than sparse geography dummies.
4. **Clustering** — elbow + silhouette over k=2..8. Production model uses four playbook segments:
   - **C3 Corporate Buyers** — all company accounts
   - **C1 / C2 / C4** — K-Means (k=3) on individuals, names assigned from cluster stats
5. **Validation** — Ward hierarchical clustering on the same feature matrix (Adjusted Rand Index vs K-Means).

## Segments

| Cluster | Buyer type | What we observe in this dataset |
| --- | --- | --- |
| C1 | Global Investors | Highest non-US share among individuals |
| C2 | First-Time Buyers | Largest home-purchase pool |
| C3 | Corporate Buyers | 103 company accounts |
| C4 | Luxury Investors | Older, higher satisfaction, ~7 units / ~$2.4M mean spend |

Full EDA, caveats, and recommendations: [`docs/research_paper.md`](docs/research_paper.md).

## Unified Mentor submission links

Copy from [`docs/SUBMISSION.md`](docs/SUBMISSION.md).

| Field | Link |
| --- | --- |
| GitHub repository | https://github.com/deepesh2804yadav/BuyerSegmentation |
| Research paper (PDF) | https://github.com/deepesh2804yadav/BuyerSegmentation/blob/main/docs/research_paper.pdf |
| Web dashboard | https://htmlpreview.github.io/?https://raw.githubusercontent.com/deepesh2804yadav/BuyerSegmentation/main/docs/dashboard.html |
| Streamlit Cloud | Deploy `app.py` from `main` at https://share.streamlit.io/ then paste the `*.streamlit.app` URL |
| Feedback video | https://github.com/deepesh2804yadav/BuyerSegmentation/blob/main/docs/project-feedback-demo.mp4 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

Writes `data/processed/client_segments.csv`, cluster metrics, joblib models, and diagnostic figures under `artifacts/`.

## Dashboard

```bash
streamlit run app.py
```

Modules:

- Buyer Segmentation Overview
- Investor Behavior Dashboard
- Geographic Buyer Analysis
- Segment Insights Panel

Sidebar filters: country, region, acquisition purpose, client type.

## Tests

```bash
python -m pytest
```
