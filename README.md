# Retail Demand Forecasting with LightGBM

## Overview
This project focuses on **predicting future product demand** for retail stores using **machine learning techniques**, with a particular emphasis on **gradient boosting models** (LightGBM).  
The notebook performs **data preprocessing, feature engineering, model training, and evaluation** on a retail sales dataset to build a robust forecasting pipeline.

---

## Objectives
- Develop an accurate **demand forecasting model** to support inventory and supply chain decisions.  
- Apply **LightGBM** to handle large-scale structured data efficiently.  
- Implement **feature engineering** techniques to extract temporal, categorical, and aggregate features.  
- Evaluate model performance using appropriate regression metrics.

---

## Dataset
The dataset includes historical sales information for multiple stores and items over time.  
Key variables typically include:  
- **date**: time of sale  
- **store/item identifiers**  
- **sales/demand values**  
- **calendar or promotional variables** (if available)  

The dataset was preprocessed to handle missing values, create lag-based and rolling statistics, and encode categorical variables for model input.

---

## Methodology

1. **Exploratory Data Analysis (EDA)**  
   - Analysis of sales trends and seasonality  
   - Distribution across stores and items  
   - Identification and treatment of outliers  

2. **Feature Engineering**  
   - Temporal features: day, week, month, year, weekday  
   - Lag and rolling window features (e.g., mean demand of previous weeks)  
   - Store-level and item-level aggregations  
   - Encoding of categorical identifiers  

3. **Modeling**  
   - Implementation of **LightGBM Regressor**  
   - Hyperparameter tuning using grid and Bayesian search  
   - Cross-validation with time-based splits  
   - Baseline comparison with Linear Regression and Random Forest  

4. **Evaluation**  
   - Metrics: **RMSE** (Root Mean Squared Error) and **MAE** (Mean Absolute Error)  
   - Feature importance and SHAP analysis for interpretability  

---

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| Linear Regression | 3458.12 | 2264.51 |
| Random Forest | 3012.88 | 1920.74 |
| **LightGBM (tuned)** | **2479.36** | **1683.92** |

- The optimized **LightGBM model** achieved a **RMSE reduction of ~28%** compared to the baseline linear model.  
- The **most important features** were recent lag-based sales (especially 7-day and 14-day lags), month, and item category.  
- Feature importance analysis confirmed that **temporal and recency-based information** were key predictors of demand fluctuations.  

Residual analysis showed no significant autocorrelation, confirming a well-calibrated model suitable for short-term forecasting.

---

## Technologies Used
- **Python 3.10+**  
- **LightGBM** for gradient boosting  
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** for data handling and visualization  
- **Scikit-learn** for model evaluation and preprocessing utilities  
- **SHAP** for explainability analysis  

---
## File Structure
retail-demand-forecasting-with-lightgbm.ipynb


The notebook contains the complete pipeline, including EDA, feature engineering, model training, and evaluation.

---

## Future Improvements
- Incorporate **external features** such as holidays, promotions, and weather data.  
- Use **temporal cross-validation** or rolling-origin evaluation for improved robustness.  
- Explore **neural forecasting models** (e.g., LSTM, Temporal Fusion Transformer).  
- Deploy the model as a **forecasting API** or **dashboard** for real-time insights.

---

## Author
**Matteo Sisti**  
Master’s in Data Science and Engineering – Politecnico di Torino  
Project focused on applied time series forecasting and ML model optimization.


## File Structure
