# Data Analysis and Clustering Project

This repository contains two complete data analysis and clustering tasks performed as part of an internship assignment.  
Both analyses were conducted on real-world marketing datasets and include detailed data preprocessing, visualization, and machine learning clustering steps.

---

##  Datasets Used

### 1. **Marketing Campaign Dataset**
**File:** `marketing_campaign.csv`  
**Description:**  
Contains demographic and purchase information for 2,240 customers. Used to identify customer segments for targeted marketing strategies.

**Key Columns:**
- `Year_Birth`, `Education`, `Marital_Status`, `Income`
- Purchase-related columns: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- Campaign acceptance columns: `AcceptedCmp1` – `AcceptedCmp5`, `Response`
- `Recency`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`

**Operations Performed:**
1. **Data Loading and Inspection** – Displayed structure, data types, and missing values.  
2. **Duplicate Handling** – Identified and removed exact and partial duplicates.  
3. **Missing Value Imputation** – Numeric columns filled using *median*, categorical using *mode*.  
4. **Feature Engineering:**
   - Created `TotalSpend` (sum of all product expenditures)
   - Created `Children` = `Kidhome` + `Teenhome`
   - Calculated `Age` = 2025 - `Year_Birth`
5. **Data Standardization** – Used `StandardScaler` for scaling numerical values.  
6. **K-Means Clustering:**
   - Determined optimal *k* using **Elbow Method** and **Silhouette Score**.
   - Chose *k = 4* for clustering.  
7. **Visualization:**
   - Elbow and Silhouette plots
   - PCA (2D projection of clusters)
   - Heatmap of cluster means  
8. **Insights & Recommendations:**
   - Interpreted cluster profiles based on income, spend, and age.  
   - Classified segments as *Premium Customers*, *Value Seekers*, *High-Income Low-Engagement*, and *Low Spenders*.

---

### 2. **Mall Customers Dataset**
**File:** `Mall_Customers.csv`  
**Description:**  
Dataset of 200 customers including gender, age, annual income, and spending score. Used for customer segmentation.

**Key Columns:**
- `CustomerID`, `Genre`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

**Operations Performed:**
1. **Data Inspection** – Checked shape, types, summary statistics, and null values.  
2. **Duplicate Removal** – Ensured unique entries.  
3. **Missing Value Imputation** – Handled missing values via *median* (numeric) and *mode* (categorical).  
4. **Encoding:** Converted `Genre` to numeric (`0` = Male, `1` = Female).  
5. **Feature Scaling:** Applied `StandardScaler`.  
6. **K-Means Clustering:**
   - Evaluated *k = 2–10* using **Elbow** and **Silhouette** methods.
   - Optimal *k* chosen automatically via highest Silhouette Score.  
7. **Visualization:**
   - PCA 2D projection with labeled centroids.
   - Pairplot to visualize feature relationships.
   - Cluster heatmap for average feature values.  
8. **Cluster Profiling & Insights:**
   - Computed mean age, income, and spending for each cluster.
   - Suggested marketing actions per segment:
     - *Premium & Engaged*: Loyalty offers and upselling.
     - *High Spend, Low Income*: Value bundles.
     - *High Income, Low Spend*: Personalized offers.
     - *Low Spenders*: Awareness and retention campaigns.

---

##  Outputs Generated

- Elbow plots to determine the optimal number of clusters.  
- Silhouette plots for cluster validation.  
- PCA scatter plots visualizing customer clusters.  
- Heatmaps and pairplots for cluster profile analysis.  
- Console outputs summarizing cluster means and marketing recommendations.  
- Final cleaned and clustered datasets saved as:  
  - `marketing_campaign_clustered.csv`  
  - `Mall_Customers_Clustered.csv`

---

##  Project Insights

The clustering analysis successfully divided customers into meaningful segments for targeted marketing strategies.  
The **Marketing Campaign dataset** revealed demographic and spending-based clusters, while the **Mall Customers dataset** highlighted patterns based on income and spending behavior.

Both analyses demonstrate the power of **unsupervised learning (K-Means)** in identifying customer behavior trends.

---


##  Tools & Libraries Used

- **Python 3**
- **Pandas**, **NumPy** – Data manipulation and analysis  
- **Matplotlib**, **Seaborn** – Visualization  
- **Scikit-learn** – Scaling, K-Means clustering, PCA, silhouette score  
- **Statsmodels** – (Used in earlier version for statistical analysis)  

---
