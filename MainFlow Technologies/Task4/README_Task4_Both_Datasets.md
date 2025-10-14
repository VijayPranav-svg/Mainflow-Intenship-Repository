
#  Regression Analysis Projects (Task 4)

##  Overview
This repository contains two complete **Regression Analysis** projects completed as part of  
**“Data Analysis and Data Science with Python – Task 4”** from Main Flow Services & Technologies Pvt. Ltd.

Both projects follow the same workflow:
- Data loading and exploration  
- Handling missing values and duplicates  
- Feature scaling and encoding  
- Correlation and outlier analysis  
- Linear Regression model training  
- Evaluation using RMSE and R²  
- Feature insights and conclusions  

---

##  Projects Included
| # | Dataset | Description | Target Variable |
|---|----------|--------------|-----------------|
| **1** |  *California Housing Dataset* | Predict median house value in California districts. | `median_house_value` |
| **2** |  *Ames Housing Dataset* | Predict sale price of residential homes in Ames, Iowa. | `SalePrice` |

---

##  Common Steps in Both Projects
### 1️ Data Loading & Exploration
### 2 Missing Value Handling
### 3️ Data Visualization
### 4️ Preprocessing
### 5️ Feature Selection
### 6️ Model Training
### 7️ Model Evaluation
### 8️ Feature Importance
### 9️ Insights & Conclusions
---
## 🏡 California Housing – Summary

**Dataset Features:**  
`longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`,  
`population`, `households`, `median_income`, `ocean_proximity`, and `median_house_value`.

**Highlights**
- Strong positive correlation: `median_income`, `housing_median_age`
- Slight negative correlation: `longitude`, `latitude`
- RMSE and R² indicate good predictive performance.
- Outliers visualized but not removed (represent luxury houses near the coast).

---

## 🏠 Ames Housing – Summary

**Dataset Features:**  
82 attributes including `Lot Area`, `Overall Qual`, `Year Built`, `Gr Liv Area`,  
`Garage Cars`, `Total Bsmt SF`, and `SalePrice`.

**Highlights**
- Top correlated predictors: `Overall Qual (0.80)`, `Gr Liv Area (0.71)`, `Garage Cars`, `Total Bsmt SF`.
- Features like `Kitchen AbvGr` and `Enclosed Porch` negatively influence prices.
- Outlier analysis performed using boxplots.
- Linear model achieved strong R² (~0.82) and reasonable RMSE (~35 000).

---

## 📊 Evaluation Metrics (Typical Results)
| Dataset | RMSE | R² |
|----------|------|----|
| **California Housing** | ~70 000 | ~0.76 |
| **Ames Housing** | ~35 000 | ~0.82 |

---

## 📈 Visualizations
- Distribution of target variable (price)
- Boxplots of key numerical features
- Correlation heatmaps
- Actual vs Predicted scatter plots

---

## 🏁 Conclusions
- Both regression models successfully predict house prices based on multiple property features.  
- Preprocessing and pipeline structure ensure reproducibility.  
- Feature importance provides clear insights into the key drivers of housing prices.  
- All steps specified in **Task 4: Regression Analysis** have been completed.

---

## 📂 Repository Structure
```
├── California_Housing_Task4.ipynb
├── Ames_Housing_Task4.ipynb
├── housing.csv
├── AmesHousing.csv
├── README.md
└── images/
    ├── correlation_heatmap_california.png
    ├── correlation_heatmap_ames.png
    ├── price_distribution.png
    ├── actual_vs_predicted.png
```

