# Data Analysis Internship Task - 2

This repository contains my submission for the **Data Analysis and Data Science Internship Task - 2** given by **Main Flow Services and Technologies Pvt. Ltd.**.

The task involved performing **Exploratory Data Analysis (EDA)** and **Sales Performance Analysis** on real-world datasets using Python.

---

##  Files

* `GlobalSuperStoreDataExp.py` → Exploratory Data Analysis on **Global Superstore dataset**.
* `AmazonDataAnalysis.py` → Sales performance analysis on **Amazon product dataset**.
* `superstore.csv` → Dataset copied from Google (Global Superstore).
* `amazon.csv` → Dataset copied from Google (Amazon products).

---

##  Project 1: General EDA

**Objective**: Explore the Global Superstore dataset.
**Steps Implemented**:

* Removed duplicates and handled missing values.
* Detected and removed outliers using Z-score.
* Computed summary statistics (mean, median, mode, std, variance).
* Analyzed correlations between features.
* Visualized data using:

  * Histograms (Sales, Profit, etc.)
  * Boxplots (outlier detection)
  * Heatmap (correlation analysis)

---

## Project 2: Sales Performance Analysis

**Objective**: Analyze sales-related data to identify factors affecting performance.

Since the given dataset (`sales_data.csv`) was **not provided**, I confirmed in the **Telegram group** that we could use our own datasets from Google. Based on that:

* I used an **Amazon products dataset** (with columns like `actual_price`, `discounted_price`, `rating`, `rating_count`, `category`).
* Data preprocessing included cleaning symbols, numeric conversion, handling missing values, and duplicate removal.
* Performed EDA with:

  * Histogram of discounted prices
  * Scatter plot of rating vs discounted price
  * Bar chart of top categories
* Built a **Linear Regression model**:

  * Target: `discounted_price`
  * Features: `actual_price`, `rating`
  * Evaluated using **R² score** and **RMSE**

---

##  Notes on Dataset Differences

The original task mentioned `sales_data.csv` with columns (`Product, Region, Sales, Profit, Discount, Category, Date`).
Since this dataset was not available, I used alternative datasets:

* Global Superstore → for Project 1 (EDA).
* Amazon products → for Project 2 (Sales Analysis & Prediction).

**Limitations due to dataset differences**:

* No `Date` column → time series sales trend was not possible.
* No `Region` column → regional sales distribution not analyzed.
* No `Profit` column → regression used `actual_price` and `rating` instead.

These changes were discussed in the Telegram group, and it was confirmed we could proceed with datasets downloaded from Google.

## For Some of the Files Due to the Space Restriction of 25MB They Have Been Converted into a ZIP and Uploaded I Kindly Request You to Extract it will Revewing
---




