# Buyer Segmentation & Investment Profiling

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=deepesh2804yadav/BuyerSegmentation&branch=main&mainModule=app.py)

Machine-learning buyer intelligence for the Parcl real estate platform, prepared for the Unified Mentor × Parcl Co. Limited market-intelligence brief.

The pipeline cleans client and listing data, engineers investment profiles, clusters buyers with K-Means (validated by hierarchical clustering), and serves a Streamlit dashboard for segment targeting.

**Live Streamlit app:** after you click the badge above (GitHub login, Python 3.12), Streamlit hosts it at a `https://….streamlit.app` URL. Suggested subdomain: [`parcl-buyer-intelligence.streamlit.app`](https://parcl-buyer-intelligence.streamlit.app).

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
| Streamlit app (`*.streamlit.app`) | [Deploy from GitHub](https://share.streamlit.io/deploy?repository=deepesh2804yadav/BuyerSegmentation&branch=main&mainModule=app.py) → then use your `https://….streamlit.app` URL (suggested: https://parcl-buyer-intelligence.streamlit.app) |
| Feedback video | https://github.com/deepesh2804yadav/BuyerSegmentation/blob/main/docs/project-feedback-demo.mp4 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements.txt` is the Streamlit Cloud install (dashboard only). Use `requirements-dev.txt` to train models and run tests.

## Deploy on Streamlit Community Cloud (`*.streamlit.app`)

The public GitHub repo is ready for [Streamlit Community Cloud](https://share.streamlit.io/). A `*.streamlit.app` URL is created only after you sign in with GitHub (Streamlit does not allow deploying someone else's account from this environment).

1. Open this deploy link while logged into the GitHub account that owns the repo:  
   https://share.streamlit.io/deploy?repository=deepesh2804yadav/BuyerSegmentation&branch=main&mainModule=app.py
2. If asked, authorize Streamlit to read `deepesh2804yadav/BuyerSegmentation`.
3. Confirm **Main file path** is `app.py` and **branch** is `main`.
4. Click **Advanced settings** and set **Python version** to **3.12**.
5. Optional: set **App URL** / custom subdomain to `parcl-buyer-intelligence` so the app is  
   `https://parcl-buyer-intelligence.streamlit.app`
6. Click **Deploy** and wait a few minutes. Use the `https://….streamlit.app` URL on the internship form.

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
