#  Retail Demand Forecasting with Machine Learning  
**Competition:** [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

---

##  Overview
This project aims to forecast **daily product sales** across thousands of Favorita grocery stores in Ecuador.  
The objective is to design a scalable forecasting framework that combines **classical time-series analysis** and **modern machine learning** techniques to capture both **temporal dependencies** and **cross-sectional relationships** among stores, product families, promotions, and external regressors.

---

## Objectives
- Predict unit sales for each `(store_nbr, family, date)` tuple in the test period.  
- Exploit **historical, promotional, transactional, and macroeconomic** data.  
- Engineer **lag and rolling window features** to capture seasonality and recent trends.  
- Benchmark **ARIMA** (classical) vs. **LightGBM/XGBoost** (ML-based) approaches.  
- Evaluate performance using **RMSLE**, the official competition metric.

---

##  Dataset
The dataset consists of multiple CSV files provided by the competition:

| File | Description |
|------|--------------|
| `train.csv` | Historical daily sales per store and family (target variable `sales`). |
| `test.csv` | Future dates for which sales must be predicted. |
| `stores.csv` | Metadata about each store (cluster, city, state, type). |
| `holidays_events.csv` | Calendar of national, regional, and local holidays (with transferred dates). |
| `oil.csv` | Daily oil price time series (macroeconomic proxy). |
| `transactions.csv` | Number of daily receipts per store (proxy for foot traffic). |

**Granularity:** Daily observations per `(store_nbr, family)`  
**Training period:** Jan 2013 → Aug 2017  
**Forecast horizon:** Aug 16 → Aug 31, 2017  

---

##  Methodology

###  Exploratory Data Analysis (EDA)
- Visualized **daily and monthly sales trends** to confirm **non-stationarity** and strong **seasonality**.  
- Identified **holiday spikes** and **promotion-driven sales** increases.  
- Explored **store-type heterogeneity** and **dominant product families** (e.g., GROCERY, BEVERAGES).

###  Data Cleaning & Integration
- **Holiday filtering:** removed “Work Day” and transferred duplicates; encoded a binary `is_holiday` flag.  
- **Oil series:** missing values imputed via **forward fill (ffill)**.  
- **Merging:** joined external regressors (holidays, oil, transactions, stores) on `date` and `store_nbr`.  
- **Calendar features:** `year`, `month`, `day`, `dayofweek`, `weekofyear`, and `is_weekend`.  
- **Memory optimization:** numeric downcasting to enable fast computation and lag generation.

###  Feature Engineering
Engineered **time-based regressors** to capture short- and long-term dependencies:

| Feature Type | Description |
|---------------|-------------|
| **Lag features** | `lag_1`, `lag_7`, `lag_14`, `lag_28`, `lag_365` — recent and annual dependencies |
| **Rolling stats** | Means and standard deviations over 7, 14, 28 days (shifted by one day to prevent leakage) |
| **Promotions & external** | `onpromotion`, `transactions`, `dcoilwtico`, `is_holiday` |
| **Calendar & events** | `month`, `dayofweek`, `is_weekend`, `is_christmas`, `is_newyear` |

Features were created **per (store_nbr, family)** group to preserve temporal structure.

###  Modeling
Compared several forecasting approaches:

| Model | Description | RMSLE |
|--------|--------------|-------|
| **Naïve / Moving Average** | Baseline reference | ~1.05 |
| **ARIMA (subset)** | Traditional time-series model | ~0.95 |
| **LightGBM (Gradient Boosting)** | ML-based with lag features | **~0.87** |

**LightGBM configuration:**
- Early stopping: 100 rounds  
- Best iteration: 426  
- Validation RMSE: 409.30  
- Validation RMSLE: 0.7953  

---

###  Feature Importance
Top predictive drivers:
1. `roll_mean_7`  
2. `lag_7`  
3. `roll_mean_28`  
4. `lag_1`  
5. `transactions`, `onpromotion`, `dcoilwtico`  

→ Confirms the dominance of **weekly seasonality** and **recent trend memory** in sales prediction.

---

###  Final Training & Submission
- Combined full training data (pre-2017-08-16) and test set to ensure consistent lag computation.  
- Retrained final LightGBM model using the **best iteration (426)**.  
- Generated forecasts for the competition’s test horizon.  
- Clipped predictions to non-negative values and exported `submission.csv` in Kaggle format.

---

##  Results Summary

| Model | RMSE | RMSLE | Rank (approx.) |
|--------|------|--------|----------------|
| Naïve / MA | — | 1.05 | — |
| ARIMA (subset) | — | 0.95 | — |
| **LightGBM (final)** | 409.30 | **0.7953** | ~468 / 662 |

 The model captures basic seasonality and promotion dynamics.  
 Still below top-performing Kaggle entries (≈ 0.50 RMSLE).  
 Future work: advanced hierarchical modeling, per-family tuning, and Bayesian hyperparameter optimization (Optuna).

---

##  Insights & Next Steps
- Weekly patterns (`lag_7`, `roll_mean_7`) dominate — **temporal context > categorical context**.  
- Limited impact of raw calendar features implies potential redundancy with rolling windows.  
- To reach top performance:
  - Add **hierarchical per-family training**.  
  - Integrate **price, oil lagged interactions**, and **store clustering embeddings**.  
  - Implement **GPU-based Optuna tuning** or **Bayesian optimization** for LightGBM.  
  - Explore **hybrid deep learning models (LSTM, TFT)** for multi-horizon forecasting.

---

##  Technologies Used
- **Python 3.10+**  
- **LightGBM** for gradient boosting  
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** for data manipulation and visualization  
- **Scikit-learn** for model evaluation and preprocessing utilities  

---

##  File Structure
retail-demand-forecasting-with-lightgbm.ipynb
The notebook contains the complete pipeline, including **EDA**, **feature engineering**, **model training**, and **evaluation**.

---

##  Future Improvements
- Incorporate **additional external features** such as detailed holidays, promotions, and weather data.  
- Use **temporal cross-validation** or **rolling-origin evaluation** for improved robustness.  
- Explore **neural forecasting models** (e.g., LSTM, Temporal Fusion Transformer).  
- Deploy the model as a **forecasting API** or **interactive dashboard** for real-time insights.

---

##  Author
**Matteo Sisti**  
MSc in Data Science and Engineering – *Politecnico di Torino*  
Project focused on **applied time series forecasting** and **ML model optimization**.  
GitHub: [snevil](https://github.com/snevil) · Kaggle: [snevil](https://www.kaggle.com/snevil)



