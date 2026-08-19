# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

**Domain:** Financial analytics and real estate market intelligence  
**Collaborating organizations:** [Unified Mentor](https://unifiedmentor.com/), [Parcl Co. Limited](https://www.parcllabs.com/)  
**Data:** 2,000 clients and 10,000 property listings (7,305 sold)

## 1. Problem

Parcl currently lacks a quantitative view of buyer types, investment motives, geography, and financing. Treating every prospect the same produces generic campaigns and weak investor targeting. This study builds unsupervised segments that marketing and sales can act on.

## 2. Data and cleaning

Clients include type (Individual / Company), gender, country, region, date of birth, acquisition purpose (Home / Investment), loan flag, referral channel, and satisfaction (1–5). Listings include unit category, floor area, sale price, status, and `client_ref`.

Cleaning steps:

- Deduplicate on `client_id`.
- Normalize labels (`corporate` → Company, mixed loan/gender strings, Home vs Personal use).
- Parse mixed `MM-DD-YYYY` / `M/D/YYYY` dates; compute age at 1 January 2024 (median fill for impossible ages).
- Parse sale prices such as `$300,385.62`.
- Aggregate sold listings per client: units purchased, total investment, average unit price, average area, office-unit share.
- Flag international buyers as country ≠ USA.

**EDA snapshot (n = 2,000):**

| Metric | Value |
| --- | --- |
| Individual / Company | 1,897 / 103 |
| Home / Investment purpose | 69.3% / 30.8% |
| Loan applied | 36.8% |
| Non-US residence | 23.1% |
| Mean age | 53.6 years (23–93) |
| Mean satisfaction | 3.03 |
| Mean units purchased | 3.65 |
| Aggregate sold volume | $2.52B |
| Top countries | USA (1,538), UK (95), Canada (85) |
| Top region | California (633) |
| Referral mix | Website 55%, Agency 35%, Client 10% |

Every client is linked to at least one sold unit. Age, loan use, and satisfaction are only weakly correlated with spend, so segments are not a simple “rich vs young” split.

## 3. Feature encoding and scaling

High-cardinality region and country one-hots flatten distances and hide investment behavior. The production feature set is therefore compact:

- Numeric (StandardScaler): age, satisfaction, units, log total investment, log average unit price, office share
- Binary: company, investment purpose, loan, international
- One-hot: referral channel
- Feature weights: company status, international flag, and log spend are up-weighted

This follows the brief (one-hot + scaling) without letting 57 regions dominate K-Means.

## 4. Clustering and cluster count

K-Means inertia and silhouette were computed for k = 2…8 on the full encoded matrix:

| k | Inertia | Silhouette |
| --- | ---: | ---: |
| 2 | 16,098 | 0.162 |
| 3 | 14,269 | **0.169** |
| 4 | 13,252 | 0.134 |
| 5 | 12,382 | 0.124 |

Unsupervised silhouette peaks at **k = 3**. The product brief specifies four named segments, including a corporate type that is only 5.2% of the file. Pure K-Means does not isolate those 103 companies.

**Production design (k = 4 playbook):**

1. Assign every Company account to **C3 Corporate Buyers**.
2. Fit K-Means with k = 3 on individuals.
3. Name those groups from cluster statistics as C1 / C2 / C4.
4. Fit Ward hierarchical clustering on the same matrix to check agreement (Adjusted Rand Index ≈ 0.29). Moderate ARI is expected: hierarchical and K-Means recover related but not identical partitions in mixed numeric/binary space.

Diagnostic figures: `artifacts/figures/elbow_method.png`, `silhouette_scores.png`, `pca_segments.png`, `hierarchical_dendrogram.png`.

## 5. Segment interpretation

| Cluster | Buyer type | n | Company | Intl. | Loan | Investment purpose | Mean age | Mean satisfaction | Mean units | Mean spend |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1 | Global Investors | 837 | 0% | 26.9% | 38.5% | 32.5% | 54.2 | 2.91 | 3.3 | $0.98M |
| C2 | First-Time Buyers | 1,011 | 0% | 20.5% | 35.1% | 28.8% | 53.5 | 3.11 | 3.7 | $1.44M |
| C3 | Corporate Buyers | 103 | 100% | 22.3% | 41.7% | 35.0% | 45.6 | 3.07 | 3.7 | $1.26M |
| C4 | Luxury Investors | 49 | 0% | 14.3% | 32.7% | 32.7% | 62.7 | 3.43 | 7.2 | $2.38M |

**C1 Global Investors.** Largest foreign-buyer share. Use for international media, currency-aware offers, and agency partners in UK, Canada, Germany, and France.

**C2 First-Time / mass home buyers.** The volume segment. Purpose is the most home-oriented. In this extract they are not distinctly younger or more leveraged than C1; the playbook name is retained, but campaigns should emphasize financing education and mid-market inventory rather than “first-time only” creative.

**C3 Corporate Buyers.** All company records. Younger named contacts, slightly higher investment purpose and loan use. Treat as relationship accounts: bulk discounts, office inventory, multi-unit packages.

**C4 Luxury Investors.** Small, older, highest satisfaction and by far the largest portfolios. Route to senior sales; protect with concierge service and scarce premium stock.

## 6. Recommendations

1. **Stop average-buyer marketing.** Split spend across C2 volume (website + mortgage partners) and C1 cross-border (agency).
2. **Create a corporate desk** for the 103 company clients; they are invisible in a single K-Means run.
3. **Whitelist C4** for off-market listings. 49 buyers concentrate high spend.
4. **Do not overfit geography.** California dominates counts because the book is US-heavy; use the dashboard region filter before concluding a local “type.”
5. **Refresh quarterly.** Re-run `python -m src.train` as listings close so luxury and corporate portfolios stay current.
6. **Collect missing features** that would sharpen first-time vs luxury: household income, prior property count, and declared budget. Age and satisfaction alone are weak separators here.

## 7. Dashboard

`streamlit run app.py` provides:

- Cluster distribution and purpose / client-type mix
- Spend, loan, and channel behavior
- Country choropleth, top regions, country × segment treemap
- Per-segment descriptive statistics and recommended actions

Filters: country, region, acquisition purpose, client type.

## 8. Conclusion

Clustering plus a corporate business rule reveals four actionable Parcl buyer groups that a single average profile conceals. Hierarchical clustering corroborates a low-to-moderate structure in the feature space; silhouette scores warn against claiming crisp, highly separated types. The value is operational: different creative, inventory, and coverage models for global, volume, corporate, and luxury demand.
