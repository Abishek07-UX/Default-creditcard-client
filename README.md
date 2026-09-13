# AIML Group Assignment: Credit Card Default Preprocessing Pipeline

[![Dataset: UCI Credit Card](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue.svg)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
[![Environment: Google Colab](https://img.shields.io/badge/Environment-Google%20Colab%20%2F%20Jupyter-orange.svg)](https://colab.research.google.com/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)]()

This repository contains the end-to-end data preprocessing pipeline and individual member notebooks for the **Default of Credit Card Clients** dataset (Taiwan, April–September 2005), completed for the AIML Group Assignment. The project implements a sequential multi-stage pipeline designed to prepare noisy, dirty, and heavy-tailed credit card behavioral data for predictive classification.

---

## 1. Dataset Overview

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Observations:** 30,000 credit card clients
- **Raw Features:** 24 features (demographic attributes, credit limits, 6-month repayment histories, 6-month bill statements, and 6-month payment amounts)
- **Target Feature:** `default payment next month` (binary: `1` = default, `0` = non-default; overall default rate: **22.12%**)
- **Data Integrity Profile:** While there are no literal missing values (`NaN`/`None`), the dataset contains undocumented/dirty categorical codes (`EDUCATION`: 0, 5, 6; `MARRIAGE`: 0), severe right-skewed monetary distributions, extreme payment spikes, and high multicollinearity across monthly bill statements.

---

## 2. Group Members & Preprocessing Assignments

Each member implemented and justified a distinct data preprocessing technique in an individual notebook, passing intermediate outputs sequentially to ensure true collaborative integration:

| Member | Student IT Number | Assigned Preprocessing Technique | Individual Notebook Filename | Input Data | Output CSV |
|---|---|---|---|---|---|
| **M1** | `IT001` | **Missing / Invalid Data Handling** | [`IT001_MissingData.ipynb`](file:///notebooks/IT001_MissingData.ipynb) | `data/raw/UCI_Credit_Card.csv` | `results/outputs/stage1_missing_handled.csv` |
| **M2** | `IT002` | **Encoding Categorical Variables** | [`IT002_Encoding.ipynb`](file:///notebooks/IT002_Encoding.ipynb) | `stage1_missing_handled.csv` | `results/outputs/stage2_encoded.csv` |
| **M3** | `IT003` | **Outlier Detection & Treatment** | [`IT003_OutlierRemoval.ipynb`](file:///notebooks/IT003_OutlierRemoval.ipynb) | `stage2_encoded.csv` | `results/outputs/stage3_outliers_removed.csv` |
| **M4** | `IT004` | **Feature Engineering — Creation** | [`IT004_FeatureEngineering_Creation.ipynb`](file:///notebooks/IT004_FeatureEngineering_Creation.ipynb) | `stage3_outliers_removed.csv` | `results/outputs/stage4_features_created.csv` |
| **M5** | `IT005` | **Normalization & Feature Scaling** | [`IT005_Scaling.ipynb`](file:///notebooks/IT005_Scaling.ipynb) | `stage4_features_created.csv` | `results/outputs/stage5_scaled.csv` |
| **M6** | `IT006` | **Feature Selection & PCA Reduction** | [`IT006_FeatureSelection_PCA.ipynb`](file:///notebooks/IT006_FeatureSelection_PCA.ipynb) | `stage5_scaled.csv` | `results/outputs/stage6_final.csv` |
| **Group** | `All Members` | **Unified Integrated Pipeline** | [`group_pipeline.ipynb`](file:///group_pipeline.ipynb) | `data/raw/UCI_Credit_Card.csv` | `results/outputs/final_processed.csv` |

*(Note: Replace placeholder IT numbers `IT001`–`IT006` with actual student IDs prior to submission).*

---

## 3. Sequential Pipeline Architecture

```mermaid
flowchart TD
    raw["data/raw/UCI_Credit_Card.csv<br/>(30,000 rows × 25 cols)"] --> S1["M1: Missing & Invalid Data Handling<br/>Remap EDUCATION (0,5,6 -> 4) & MARRIAGE (0 -> 3)"]
    S1 --> O1["results/outputs/stage1_missing_handled.csv"]
    O1 --> S2["M2: Categorical Encoding<br/>Binary (SEX_FEMALE), One-Hot (EDUCATION, MARRIAGE)"]
    S2 --> O2["results/outputs/stage2_encoded.csv"]
    O2 --> S3["M3: Outlier Treatment<br/>Winsorization (1st-99th percentile) across 14 financial features"]
    S3 --> O3["results/outputs/stage3_outliers_removed.csv"]
    O3 --> S4["M4: Behavioral Feature Creation<br/>avg_bill, avg_pay, credit_utilization, payment ratios, max_delay"]
    S4 --> O4["results/outputs/stage4_features_created.csv"]
    O4 --> S5["M5: Normalization & Scaling<br/>StandardScaler (Z-Score) on continuous features"]
    S5 --> O5["results/outputs/stage5_scaled.csv"]
    O5 --> S6["M6: Selection & Dimensionality Reduction<br/>PCA on BILL_AMT1..6 (2 PCs retain >96% variance)"]
    S6 --> O6["results/outputs/stage6_final.csv"]
    
    S1 -.-> GP["group_pipeline.ipynb<br/>Modular end-to-end Python pipeline"]
    S2 -.-> GP
    S3 -.-> GP
    S4 -.-> GP
    S5 -.-> GP
    S6 -.-> GP
    GP --> FINAL["results/outputs/final_processed.csv<br/>(30,000 rows × 39 optimized cols)"]
```

---

## 4. Preprocessing Stages & Domain Justifications

### Stage 1 (M1): Missing & Invalid Data Handling
- **Problem:** Undocumented codes `0, 5, 6` in `EDUCATION` (345 rows) and `0` in `MARRIAGE` (54 rows).
- **Strategy:** Re-coded undocumented categories into the "Others" category (`EDUCATION = 4`, `MARRIAGE = 3`).
- **Why First:** Prevents the generation of spurious dummy columns or invalid arithmetic rank weights. Preserves all 30,000 rows without discarding defaulting clients.
- **EDA Plot:** [`m1_invalid_codes_distribution.png`](file:///results/eda_visualizations/m1_invalid_codes_distribution.png)

### Stage 2 (M2): Categorical Variable Encoding
- **Problem:** Categorical columns have no true linear metric spacing. Graduate School (`1`) is smaller than High School (`3`), which would distort linear regressors.
- **Strategy:** Binary indicator for `SEX` (`SEX_FEMALE`), One-Hot Encoding for `EDUCATION` (`EDUCATION_1..4`) and `MARRIAGE` (`MARRIAGE_1..3`).
- **EDA Plot:** [`m2_categorical_default_rates.png`](file:///results/eda_visualizations/m2_categorical_default_rates.png)

### Stage 3 (M3): Outlier Detection & Treatment
- **Problem:** Balance amounts and payments have extreme positive/negative anomalies (e.g. `PAY_AMT1` up to NT$ 873,552 vs 99th percentile of NT$ 66,111). Hard IQR deletion would drop >25% of all rows, severely under-sampling defaulters.
- **Strategy:** Winsorization / Soft Capping at the 1st and 99th percentiles across all 14 continuous financial features.
- **Result:** Zero data loss (30,000 rows retained) while bounding distortion on model gradients.
- **EDA Plot:** [`m3_outlier_boxplots.png`](file:///results/eda_visualizations/m3_outlier_boxplots.png)

### Stage 4 (M4): Feature Engineering — Creation
- **Problem:** Static monthly balances do not reveal underlying financial stress or cash flow habits.
- **Strategy:** Derived 13 behavioral features:
  - `avg_bill_amt` & `avg_pay_amt`: Smoothed expenditure and liquidity.
  - `credit_utilization`: $\frac{\text{avg\_bill\_amt}}{\text{LIMIT\_BAL}}$ (classic debt strain metric).
  - `payment_ratio_1..6` & `avg_payment_ratio`: Payment-to-bill fulfillment ratios.
  - `bill_to_limit_ratio`: Recent credit exposure.
  - `max_delay` & `delay_count`: Delinquency severity metrics.
- **EDA Plot:** [`m4_credit_utilization_default.png`](file:///results/eda_visualizations/m4_credit_utilization_default.png)

### Stage 5 (M5): Normalization & Feature Scaling
- **Problem:** Credit limits (up to NT$ 500,000+) dwarf unit ratios ($0.0$ to $5.0$). Unscaled features bias distance-based metrics (KNN, SVM, PCA) and gradient descent algorithms.
- **Strategy:** Standardized all 24 continuous features using `StandardScaler` ($z = \frac{x - \mu}{\sigma}$). Categorical binary indicators remain unscaled.
- **EDA Plot:** [`m5_scaling_comparison.png`](file:///results/eda_visualizations/m5_scaling_comparison.png)

### Stage 6 (M6): Feature Selection & Dimensionality Reduction
- **Problem:** `BILL_AMT1` through `BILL_AMT6` have pairwise correlations exceeding $0.85$–$0.95$ (extreme multicollinearity).
- **Strategy:** 
  - **PCA**: Compressed the 6 bill statement columns into 2 principal components (`PC1_BILL`, `PC2_BILL`) capturing **96.03%** of total bill variance.
  - **Feature Selection**: Ranked features by Pearson correlation against `default payment next month`, verifying that engineered delay and utilization metrics deliver the strongest predictive signals.
- **EDA Plot:** [`m6_pca_variance_heatmap.png`](file:///results/eda_visualizations/m6_pca_variance_heatmap.png)

---

## 5. Repository Structure

```
Default-creditcard-client/
├── README.md                                    # Comprehensive assignment documentation
├── PROJECT_PLAN.md                              # Original project brief and grading rubrics
├── UCI_Credit_Card.xls                          # Original raw Excel dataset
├── group_pipeline.ipynb                         # Integrated end-to-end master pipeline
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv                  # Converted clean CSV source (30,000 rows × 25 cols)
│   └── external/                                # External reference directory
├── notebooks/
│   ├── IT001_MissingData.ipynb                  # Stage 1: Missing & Invalid Data Handling (M1)
│   ├── IT002_Encoding.ipynb                     # Stage 2: Categorical Variable Encoding (M2)
│   ├── IT003_OutlierRemoval.ipynb               # Stage 3: Outlier Treatment & Winsorization (M3)
│   ├── IT004_FeatureEngineering_Creation.ipynb  # Stage 4: Behavioral Feature Engineering (M4)
│   ├── IT005_Scaling.ipynb                      # Stage 5: Feature Normalization & Standardization (M5)
│   └── IT006_FeatureSelection_PCA.ipynb         # Stage 6: Feature Selection & PCA Reduction (M6)
└── results/
    ├── eda_visualizations/                      # High-resolution exported EDA figures
    │   ├── m1_invalid_codes_distribution.png
    │   ├── m2_categorical_default_rates.png
    │   ├── m3_outlier_boxplots.png
    │   ├── m4_credit_utilization_default.png
    │   ├── m5_scaling_comparison.png
    │   └── m6_pca_variance_heatmap.png
    ├── logs/                                    # Execution and audit logs
    └── outputs/                                 # Intermediate and final CSV artifacts
        ├── stage1_missing_handled.csv
        ├── stage2_encoded.csv
        ├── stage3_outliers_removed.csv
        ├── stage4_features_created.csv
        ├── stage5_scaled.csv
        ├── stage6_final.csv
        └── final_processed.csv                  # Ready-to-model final dataset
```

---

## 6. How to Run

### Method A: Running in Google Colab (Recommended)
1. Upload the entire project folder to your Google Drive (e.g. `MyDrive/Default-creditcard-client`).
2. Open Google Colab and mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   os.chdir('/content/drive/MyDrive/Default-creditcard-client')
   ```
3. To execute the combined pipeline: Open and run [`group_pipeline.ipynb`](file:///group_pipeline.ipynb).
4. To inspect or grade individual work: Open and run the notebooks in sequential order:
   - `IT001_MissingData.ipynb`
   - `IT002_Encoding.ipynb`
   - `IT003_OutlierRemoval.ipynb`
   - `IT004_FeatureEngineering_Creation.ipynb`
   - `IT005_Scaling.ipynb`
   - `IT006_FeatureSelection_PCA.ipynb`

### Method B: Running Locally via Jupyter / Python
1. Ensure dependencies are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Launch Jupyter Lab / Notebook:
   ```bash
   jupyter lab
   ```
3. Open and run `group_pipeline.ipynb` or any individual notebook in `notebooks/`.
