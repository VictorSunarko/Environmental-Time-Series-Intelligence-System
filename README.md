# Environmental Time-Series Intelligence System

> **A forecasting and risk intelligence pipeline for urban air quality**  
> Beijing Multi-Site PM2.5 | 2013–2017 | 12 Monitoring Stations  
> by Victor Sunarko on 2026

---

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dashboard](#dashboard)
- [Limitations](#limitations)
- [Future Extensions](#future-extensions)
- [Citation](#citation)

---

## Overview

This project constructs a **comprehensive environmental forecasting and risk intelligence system** applied to urban air quality in Beijing, China. The pipeline integrates 18 analytical chapters covering temporal data validation, statistical inference, classical time-series modeling, machine learning forecasting, uncertainty quantification, model interpretability, regime analysis, and an AQI-based risk intelligence framework.

The project demonstrates that rigorous statistical methodology and modern machine learning are complementary, not competing, tools. Classical models (ARIMA, SARIMA, SARIMAX, GARCH) provide testable coefficients and analytical prediction intervals. Ensemble ML models (Random Forest, XGBoost, LightGBM) capture nonlinear threshold effects and complex feature interactions. SHAP bridges both worlds with theoretically grounded attribution.

**This is not a tutorial reproduction.** Every modeling decision is driven by evidence from prior analytical steps, every assumption is formally tested, and every limitation is explicitly documented.

---

## Research Question

> *"Can we model, understand, and forecast the dynamic behavior of PM2.5 concentration in Beijing in a way that is statistically sound, predictive, interpretable, and practically actionable?"*

Secondary questions:
- Do meteorological variables have nonlinear, threshold-based relationships with PM2.5 that linear models miss?
- Did China's 2013 Air Pollution Action Plan produce statistically significant structural changes in pollution dynamics?
- How well does a model trained on one urban station generalize to other stations without retraining?

---

## Dataset

### Primary: Beijing Multi-Site Air Quality (UCI ML Repository)

| Property | Value |
|---|---|
| **Source** | Beijing Municipal Environmental Monitoring Center |
| **Repository** | UCI ML Repository · DOI: 10.24432/C5RK5G |
| **Coverage** | March 1, 2013 to February 28, 2017 |
| **Stations** | 12 nationally-controlled monitoring sites |
| **Frequency** | Hourly |
| **Total observations** | 420,768 (12 stations x 35064 hours) |
| **License** | CC BY 4.0 |

**Variables:**
- **Pollutants (6):** PM2.5 *(primary target)*, PM10, SO2, NO2, CO, O3
- **Meteorological (6):** TEMP, PRES, DEWP, RAIN, WSPM, Wind Direction

**Related Introductory Paper**
This dataset is accompanied by an academic study that provides context on Beijing’s air quality dynamics, policy interventions, and observed trends during the data collection period.
**Reference (APA style):**
> Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality improvement in Beijing. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205), 20170457. https://doi.org/10.1098/rspa.2017.0457

**Citation:**
> Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G

### Extension: Jakarta Air Quality (AQICN Historical Database)

| Property | Value |
|---|---|
| **Source** | World Air Quality Index (AQICN) |
| **Coverage** | 2018 to 2026 |
| **Observations** | 2,259 daily records |
| **Variable** | PM2.5 AQI (US EPA standard) |
| **Purpose** | Comparative profile analysis (Beijing vs Jakarta) |

---

## Methodology

The project follows a **hierarchical research workflow** with 18 chapters, each building on validated outputs from preceding chapters.

```
Chapter 0-1   :  Setup, reproducibility, data acquisition
Chapter 2-4   :  Validation, missingness analysis, preprocessing
Chapter 5-6   :  Deep EDA, stationarity & time-series diagnostics
Chapter 7-8   :  Feature engineering (84 features), temporal splitting
Chapter 9     :  Baseline models (9 benchmarks)
Chapter 10    :  Classical statistical models (ARIMA → SARIMAX → GARCH)
Chapter 11    :  Machine learning models (RF, XGBoost, LightGBM)
Chapter 12    :  Uncertainty quantification (3 methods, calibration testing)
Chapter 13    :  Master evaluation & Diebold-Mariano significance testing
Chapter 14    :  SHAP interpretability (summary, dependence, interactions)
Chapter 15    :  Regime & structural analysis (Chow, CUSUM, policy effects)
Chapter 16    :  Risk intelligence framework (AQI categorization, exceedance)
Chapter 17    :  Geographic extension (cross-station transfer, Jakarta)
Chapter 18    :  Conclusions, limitations, future extensions
```

### Feature Engineering Highlights

84 features constructed across 7 evidence-based categories:

| Category | Count | Evidence Base |
|---|---|---|
| PM2.5 lag features (t−1 to t−168) | 9 | PACF cutoff at lag 2, ACF at lag 24/168 |
| Co-pollutant lags (PM10, CO, NO2) | 6 | Co-occurrence lift ratios >5x |
| Rolling statistics (6h-72h) | 19 | ACF slow decay through lag 72h |
| Cyclical time encoding (sin/cos) | 8 | Hour×Month multiplicative seasonality |
| Meteorological lags (CCF-guided) | 24 | CCF: WSPM peak at lag=3h (r=−0.307) |
| Interaction features | 6 | Accumulation index, combustion composite |
| Trend & calendar features | 5 | KPSS trend-stationarity for 5/6 pollutants |

### Validation Discipline

- **Temporal holdout split**: Train (70%); Validation (15%); and Test (15%)
- **Walk-forward CV**: 5 expanding folds for hyperparameter tuning (no random CV)
- **Test set used exactly once**: In Chapter 13 final evaluation only
- **Leakage audit**: Formal verification that all features use only past information

---

## Key Findings

### 1. Near-Random-Walk Dominance
PM2.5 at the hourly scale has lag-1 PACF = 0.969, meaning it behaves nearly as a random walk. PM25_lag_1h alone accounts for Mean |SHAP| = 60.01 µg/m³, which is 20 times larger than the next most important feature. This explains why the Naive (t−1) baseline achieves R² = 0.950 and why it is the hardest baseline to beat.

### 2. Humidity Is the Primary Meteorological Driver
Dew point temperature (DEWP) ranks as the most important meteorological feature (Mean |SHAP| = 3.03), ahead of wind speed and temperature. Stagnant, humid air traps PM2.5 via boundary layer suppression and aerosol hygroscopic growth, a physically interpretable mechanism confirmed by both SARIMAX coefficients (DEWP: β = +2.36, p < 0.001) and SHAP dependence plots.

### 3. Wind Speed Has a Nonlinear Threshold Effect (~3 m/s)
SARIMAX found WSPM insignificant (p = 0.674) in the linear model. SHAP dependence plots revealed why. Wind speed has virtually no dispersal effect below approximately 3 m/s, then triggers a sharp negative PM2.5 response above this threshold. Linear models cannot detect this behavior, so nonlinear tree-based modeling with SHAP analysis is required.

### 4. Policy-Driven Structural Decline Is Statistically Confirmed
Year-over-year PM2.5 declines are significant: 2014 to 2015 (−10.2%), 2015 to 2016 (−9.6%), and 2013 to 2016 (−10.2%), all with Mann-Whitney p < 0.001. The Chow structural break test identified November to December 2016 as the primary break point, coinciding with the first heating season under strengthened emission limits. This validates the observable policy effect of China’s Air Pollution Action Plan.

### 5. Zero-Shot Cross-Station Transfer Works Remarkably Well
The Aotizhongxin-trained LightGBM model transferred to Dongsi (urban) with **better** performance than a station-specific retrained model (zero-shot MAE=10.99 vs retrained MAE=12.29). For the rural Dingling station, the transfer penalty was only +0.41 µg/m³ despite a 20 µg/m³ mean PM2.5 difference between stations.

### 6. Winter Pollution Is Twice as Volatile as Summer
Winter standard deviation divided by summer standard deviation equals 2.06. This season-dependent heteroskedasticity explains the 95% PI under-coverage in uncertainty quantification, with 84.9% empirical coverage versus nominal 95%. The models were calibrated on a mixture of seasons but tested predominantly on winter, which operates under a fundamentally different variance regime.

### 7. Beijing and Jakarta Have Opposite Seasonality
Beijing peaks in winter (Dec–Feb, coal heating + temperature inversion). Jakarta peaks in the dry season (Jun–Oct, biomass burning + vehicle emissions). Both cities substantially exceed the WHO PM2.5 guideline (5 µg/m³). The pipeline methodology transfers to Jakarta conceptually but requires feature adaptations: biomass burning index, monsoon indicators, and different lag structure for tropical meteorology.

---

## Model Performance

### Test Set Results (Aug 2016 – Feb 2017, Aotizhongxin)

| Rank | Model | Category | MAE (µg/m³) | RMSE | R² |
|---|---|---|---|---|---|
| 1 | **LightGBM** | ML | **9.96** | 18.27 | **0.9623** |
| 2 | Random Forest | ML | 10.28 | 19.15 | 0.9585 |
| 3 | XGBoost | ML | 10.52 | 18.87 | 0.9598 |
| 4 | **Naive (t-1)** | Baseline | **10.98** | 20.94 | 0.9504 |
| 5 | Rolling Mean (6h) | Baseline | 22.94 | 40.20 | 0.8174 |
| 6 | Rolling Mean (12h) | Baseline | 31.43 | 52.40 | 0.6900 |
| … | … | … | … | … | … |
| 13 | SARIMAX** | Classical | 95.83 | 137.76 | −1.157 |

> ** SARIMAX performance reflects 5,110-step multi-step compounding error, which is a fundamentally different forecasting task from the 1-step-ahead ML models. Its value lies in coefficient inference and analytical prediction intervals, not in point accuracy.

### Diebold-Mariano Tests (ML vs Naive Baseline)

| Comparison | DM Statistic | p-value | Result |
|---|---|---|---|
| Naive vs Random Forest | 5.36 | <0.001 | **Significant** |
| Naive vs XGBoost | 3.82 | 0.0001 | **Significant** |
| Naive vs LightGBM | 5.33 | <0.001 | **Significant** |

### Risk Intelligence Performance

| Metric | Value |
|---|---|
| Exact AQI category accuracy | 80.8% |
| Within-one-category accuracy | **99.1%** |
| Mean P(PM2.5 > China Std 75) | 40.7% |
| Mean P(PM2.5 > Hazardous 150.4) | 19.6% |
| LightGBM training time | **0.7 seconds** |

---

## Project Structure

```
environmental-tsis/
│
├── notebook/
│   └── environmental_tsis.ipynb      # Master notebook (18 chapters, 350+ cells)
│
├── dashboard/
│   └── app.py                        # Streamlit risk intelligence dashboard (upcoming)
│
├── models/
│   └── saved_models/
│       ├── arima_model.pkl           # ARIMA(3,0,1)
│       ├── sarima_model.pkl          # SARIMA(3,0,1)×(1,0,1)[24]
│       ├── sarimax_model.pkl         # SARIMAX with WSPM, TEMP, DEWP
│       ├── garch_model.pkl           # GARCH(1,1) conditional variance
│       ├── rf_model.pkl              # Random Forest (tuned)
│       ├── xgb_model.pkl             # XGBoost (early stopping)
│       └── lgb_model.pkl             # LightGBM ← best model
│
├── data/
│   ├── raw/beijing/                  # 12 original station CSVs (PRSA_Data_*.csv)
│   ├── raw/jakarta/                  # Jakarta AQICN daily CSV
│   └── processed/                    # Cleaned, imputed, feature-engineered data
│       ├── beijing_clean.csv
│       ├── feature_matrix_aotizhongxin.csv
│       ├── split_train.csv
│       ├── split_val.csv
│       └── split_test.csv
│
├── outputs/
│   ├── figures/                      # All exported plots (ch2_ to ch17_)
│   └── tables/                       # All exported CSVs and JSONs
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Prerequisites

```bash
# Python 3.11 recommended (most stable for ML libraries)
pip install -r requirements.txt
```

### Data Setup

1. Download the 12 Beijing station CSVs from [UCI ML Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) or uploading the dataset to Kaggle
2. Place all 12 `PRSA_Data_*.csv` files in `data/raw/beijing/`
3. *(Optional)* Download Jakarta historical CSV from [AQICN Historical](https://aqicn.org/historical/) and place in `data/raw/jakarta/jakarta_aqicn.csv`

### Running the Notebook

```python
# In Cell 1.1: set your environment
ENV = 'local'
DATA_DIR = r"path/to/your/data/folder"  # Update this path

# Then: Run All Cells (Run All, or Shift+Enter sequentially)
```

### Environment Switch (Kaggle/Colab)

Upload the 12 CSVs as a Kaggle dataset, then:
```python
ENV = 'kaggle'
DATA_DIR = "/kaggle/input/beijing-multisite-air-quality"
```

Everything else runs without modification.

### Reproducibility

All random seeds are fixed at `RANDOM_SEED = 42` throughout. Library versions are logged in Cell 0.5. To reproduce exactly:

```bash
pip install -r requirements.txt
# Run notebook from Cell 0.0 (installation) until Cell 18.8 (final)
```

---

## Dashboard (Upcoming)

The Streamlit dashboard provides interactive real-time risk monitoring visualization using the saved LightGBM and quantile model artifacts.

### Features
- **Forecast view**: Historical PM2.5 + 1-hour-ahead prediction with 80% PI bands
- **Risk indicator**: Color-coded AQI category badge (Good to Hazardous)
- **Exceedance probability**: P(PM2.5 > WHO / China Standard / Hazardous)
- **Station selector**: Switch between all 12 Beijing stations
- **Risk calendar**: Monthly heatmap of predicted risk days

### Running Locally

```bash
cd dashboard
streamlit run app.py
```

### Deployed Version

**Live dashboard:** `[streamlit-link-coming-soon]`  

---

## Limitations

Six key limitations are documented which each traced to specific evidence in the notebook:

1. **Near-random-walk dominance**: PM25_lag_1h (r=0.969) dominates all predictions. For genuine multi-step forecasting (24h+ without recent observations), performance will degrade substantially toward Rolling Mean or Seasonal Naive level.

2. **Test set seasonal imbalance**: Spring is absent from the test set (0% coverage); test set is 42.6% Winter. Performance in Spring and Summer estimated from validation set only.

3. **95% PI under-coverage**: LightGBM quantile PI achieves 84.9% empirical coverage vs nominal 95%. Root cause: aleatoric uncertainty dominates (inherent data randomness), not epistemic, indicates the quantile model captures the central distribution well but is overconfident at extreme tails.

4. **Structural non-stationarity**: PM2.5 declined ~10% per year from 2014–2016 (policy-driven). Models trained on 2013–2015 data and applied post-2016 will increasingly reflect historical rather than current dynamics without periodic retraining.

5. **Single-station modeling**: Main pipeline uses Aotizhongxin only. The 12-station spatial correlation structure (mean inter-station r=0.885) is unused. A joint spatial-temporal model would capture this signal.

6. **Data age**: Dataset covers 2013–2017. China's air quality dynamics have changed substantially since then (2018 Blue Sky Plan, COVID-19 effects, heating electrification). Validation against post-2018 data is recommended before any operational deployment.

---

## Future Extensions

**Tier 1, High value, feasible:**
- **Conformal prediction intervals**: provably valid coverage without distributional assumptions (directly addresses 95% PI under-coverage)
- **Multi-step ahead forecasting**: reformulate targets as y_{t+24} and y_{t+48}, eliminating lag-1 access
- **Spatial panel model**: exploit inter-station r=0.885 via Graph Neural Networks or spatiotemporal kriging

**Tier 2, Research-grade:**
- **Temporal Fusion Transformer**: simultaneous multi-horizon outputs, long-range temporal dependencies
- **Jakarta full pipeline**: adapted feature set for tropical meteorology and biomass burning
- **Causal inference for policy evaluation**: use Nov 2016 structural break as natural experiment in difference-in-differences framework

---

## Citation

If you use this project or its methodology:

```bibtex
@misc{sunarko2026environmental,
  author    = {Victor Sunarko},
  title     = {Environmental Time-Series Intelligence System:
                A Forecasting and Risk Intelligence Pipeline for Urban Air Quality},
  year      = {2026},
}
```

**Dataset citation:**
> Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G.

---

## Tech Stack

| Purpose | Libraries |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Statistical tests | `statsmodels`, `scipy`, `arch` |
| Time-series models | `statsmodels` (ARIMA/SARIMA/SARIMAX), `arch` (GARCH) |
| Machine learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Interpretability | `shap` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dashboard | `streamlit` |
| Utilities | `joblib`, `tqdm`, `ucimlrepo` |

---

<div align="center">

*"A forecast without uncertainty is not a forecast, it is an assertion."*

**Victor Sunarko · 2026**

[🖥️ Dashboard - Soon](upcoming-streamlit-link)

</div>
