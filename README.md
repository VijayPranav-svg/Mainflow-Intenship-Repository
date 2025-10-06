# Mainflow-Intenship-Repository
All The Codes and  Scripts Regarding the Internship Tasks  are placed in this repository with their Respective Task Order


#  Data Analysis with Python – Task 1 Overview

This repository contains my **internship project submission** for **Data Analysis and Data Science **.  
##  Tasks Overview

###  [Task 1 – Data Analysis with Python](https://github.com/VijayPranav-svg/Mainflow-Intenship-Repository/tree/main/MainFlow%20Technologies/Task1)
**Objective:** Perform exploratory data analysis (EDA) and visualization using Python libraries.  
**Key Steps:**
- Data cleaning and preprocessing  
- Statistical summaries and visualization  
- Correlation and feature distribution analysis  
- Insights derived using Matplotlib and Seaborn  

**Status:**  *Completed*

---

###  [Task 2 – Global Superstore Data Analysis](https://github.com/VijayPranav-svg/Mainflow-Intenship-Repository/tree/main/MainFlow%20Technologies/Task2)
**Objective:** Analyze the Global Superstore dataset to identify profit and sales trends by region, category, and customer segment.  

**Issue & Resolution:**  
The official dataset (`Global Superstore.csv`) mentioned in the task PDF was unavailable.  
After consulting my mentor, I used a similar dataset sourced from Google for the same analysis structure.  
A **Word file explanation** of this change is attached in the Task 2 folder.

**Key Operations:**
- Data loading, null handling, and feature correction  
- Region-wise and category-wise sales and profit analysis  
- Visualization using Seaborn and Matplotlib  
- Derived insights for management-level reporting  

**Status:**  *Completed with mentor-approved dataset substitution*

---

###  [Task 3 – Customer Segmentation using K-Means Clustering](https://github.com/VijayPranav-svg/Mainflow-Intenship-Repository/tree/main/MainFlow%20Technologies/Task3)
**Objective:** Apply unsupervised learning (K-Means Clustering) to segment customers based on behavioral and demographic patterns.

**Datasets Used:**
1. **Marketing Campaign Dataset** (`marketing_campaign.csv`)
   - Contains demographic and purchasing data for 2,240 customers.
   - Engineered new features:
     - `TotalSpend`, `Children`, `Age`
   - Cleaned missing data and standardized features.
   - Determined optimal clusters using:
     - **Elbow Method**
     - **Silhouette Score**
   - Selected `k = 4` as optimal.
   - Visualized results using PCA 2D projection and cluster heatmaps.
   - Interpreted segments such as:
     - *Premium Customers*
     - *Value Seekers*
     - *High-Income Low-Engagement*
     - *Low Spenders*

2. **Mall Customers Dataset** (`Mall_Customers.csv`)
   - Used gender, age, income, and spending score to group customers.
   - Encoded categorical features, standardized data, and ran K-Means clustering.
   - Automatically selected `k` using silhouette maximization.
   - Visualized results:
     - Elbow & Silhouette plots
     - PCA 2D projection with centroids
     - Cluster-wise pairplots and heatmaps  
   - Derived practical insights for marketing strategy.

**Libraries Used:**
- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn` (StandardScaler, KMeans, PCA, silhouette_score)

**Outputs Generated:**
- Cluster visualization plots
- Heatmaps & PCA projections
- Recommendations for marketing focus by customer type  

**Status:**  *Completed successfully with full visual and analytical report*

## Tasks
- [Task 1 – Data Analysis with Python]([Task1/README.md](https://github.com/VijayPranav-svg/Mainflow-Intenship-Repository/tree/main/MainFlow%20Technologies/Task1))
- [Task 2 – Global Superstore Analysis](Task2/README.md)
- [Task 3 – Customer Segmentation using K-Means Clustering](Task3/README.md)
- [Task 4 – Task 4 Title](Task4/README.md)

## Repository Structure
 Mainflow-Internship-Repository/
  - Task1/
    - code/
    - data/
    - README.md
  - Task2/
    - code/
    - data/
    - README.md
  - Task3/
    - ...
    - README.md
  - Task4/
    - ...
    - README.md
  - README.md (Main overview)

##  Current Status
-  **Task 1 completed** successfully.  
-  **Task 2 in completed** – however, the dataset download link (Global Superstore / `sales_data.csv`) is missing from the provided PDF.
-  **Task 3 in completed**  Customer Segmentation using K-Means But used My own Datasets as they are no Datasets attached with the Task Pdf Provided
-  After raising the issue, I was instructed by the internship mentor to choose my own dataset from Google and proceed with the analysis.

Therefore, due to the lack of the official dataset and with permission from the mentor, I have downloaded a dataset from Google and used it for Task 2.
I Have Also Attached A Word File Regarding this Issue For You Clear Understanding
