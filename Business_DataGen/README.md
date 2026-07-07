# Module 1: Business Understanding & Data Generation

## 1. Module Overview & Objectives
This module serves as the foundational business analysis and data engineering phase for the Real Estate Valuation project. The primary objectives are to establish the core business context, define evaluation metrics, and generate a unified dataset. 

By fusing the original Ames Housing dataset (2930 observations) <sup>[[1]](#ref1)</sup> with custom-engineered contextual features (capturing macroeconomic conditions, spatial proximity, and market liquidity), this module provides the single source of truth for the entire analytics pipeline.

## 2. Business Problem & KPIs

**1. Business Problem:**
Real estate listing platforms, buyers, and sellers all need a fast, defensible estimate of a property's fair market value prior to a formal appraisal. The core business question this project aims to answer is: *"Given a property’s characteristics and location, what is its estimated market price, and how confident is that estimate?"* Solving this problem will help the platform provide reliable pricing benchmarks, thereby facilitating smoother transactions.

**2. Key KPIs & Business Costs:**
According to the project requirements, the KPIs must address the business cost of over/under-valuation:
*   **Business Cost of Over-valuation (Predicting too high):** An inaccurately inflated estimate scares away serious buyers. This leads to properties staying on the market longer (increased days-on-market), causing frustration for sellers and reducing trust in the platform's accuracy.
*   **Business Cost of Under-valuation (Predicting too low):** An artificially low estimate "leaves money on the table for the seller". Sellers may refuse to list their properties on the platform if they feel the algorithm is undervaluing their assets.
*   **Prediction Error Tolerance:** Aim for an acceptable margin of error where a high percentage (e.g., 80-90%) of the model's predictions fall within ±10% of the actual closing sale price.

**3. Evaluation Metrics:**
To measure the success of the models and align with the business KPIs, the modeling team is required to use the following regression metrics and techniques:
*   **RMSE (Root Mean Squared Error) & MAE (Mean Absolute Error):** To measure the average magnitude of the pricing errors in dollars.
*   **MAPE (Mean Absolute Percentage Error):** To translate errors into percentage terms (e.g., "The model is off by 5% on average"), which directly ties back to the Prediction Error Tolerance KPI.
*   **R² (R-squared):** To understand the proportion of the variance in sale prices that the model can explain.
*   **Model Validation:** The model must be validated using k-fold cross-validation.
*   **Residual Analysis:** Analyze residuals specifically to identify if the model has a systematic bias toward under-prediction or over-prediction.

## 3. Module Structure
```text
.
├── Business_DataGen/
│   ├── 01_dataset_construction.ipynb
│   ├── DATA_DICTIONARY.md
│   └── README.md
│
└── data/
    ├── raw_data/
    │   ├── AmesHousing.csv
    │   └── MortgageRates.csv
    └── housing_initial_data.csv
```
The project separates data storage from code execution:
- **`Business_DataGen/01_dataset_construction.ipynb`**: The Python execution pipeline. It ingests the two raw CSV files from the `data/raw_data/` directory, maps historical U.S. 30-year fixed mortgage rates, and simulates synthetic features (DOM, Distance to Center, etc.) using documented mathematical functions.
- **`Business_DataGen/DATA_DICTIONARY.md`**: The complete dataset lexicon detailing all 87 finalized features, defining their data types, expected operational ranges, and the business assumptions behind the synthetic generation logic.
- **`data/housing_initial_data.csv`**: The finalized flat file containing the merged base and synthetic features, exported here for direct consumption by downstream modules.

## REFERENCES
<a id="ref1"></a>**[1]** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*. Journal of Statistics Education, 19(3). 
