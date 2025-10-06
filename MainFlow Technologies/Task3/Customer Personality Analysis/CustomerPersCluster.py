import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns



MarketCampsdt = pd.read_csv('marketing_campaign.csv', sep='\t')
MarketCampsdt.columns = MarketCampsdt.columns.str.strip()

# Basic inspection
print(MarketCampsdt.head().to_string())
print("\n")
print(MarketCampsdt.info())
print("\n")
print(MarketCampsdt.shape)
print("\n")
print("Data Shape:", MarketCampsdt.shape)
print("\nData Types:\n", MarketCampsdt.dtypes)
print("\nMissing Values:\n", MarketCampsdt.isnull().sum())
print("\nSummary Statistics Of Data:\n", MarketCampsdt.describe(include='all'))

DuplicateRowsum = MarketCampsdt.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", MarketCampsdt[MarketCampsdt.duplicated()])
print("\n")
database_withDuplicatedRowsGone = MarketCampsdt.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")

Partial_Duplicates_Rows_Count = MarketCampsdt.duplicated(keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
MarketCampsdt_no_partial_duplicates = MarketCampsdt.drop_duplicates(subset = MarketCampsdt.columns.difference(['Index']) , keep = 'first')
print(MarketCampsdt_no_partial_duplicates.head().to_string())

print("Missing Values in each Column: \n", MarketCampsdt.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (MarketCampsdt.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': MarketCampsdt.isnull().any(),
    'Missing_Values_Count': MarketCampsdt.isnull().sum(),
    'Missing_ValuesIn_Percentage': (MarketCampsdt.isnull().sum() / len(MarketCampsdt)) * 100
})

print(Missing_Values_Coloums_Dataframe.to_string())


def Imputation_Of_Values(database):
    for column in database.columns:
        if np.issubdtype(database[column].dtype, np.number):
            median_value = database[column].median()
            database[column] = database[column].fillna(median_value)
            print(f"Filled NaN in numeric column '{column}' with Median = {median_value}")
        
        elif database[column].dtype == 'object':
            mode_value = database[column].mode()[0]
            database[column] = database[column].fillna(mode_value)
            print(f"Filled NaN in categorical column '{column}' with Mode = {mode_value}")
        
        elif np.issubdtype(database[column].dtype, np.datetime64):
            database[column] = database[column].interpolate(method='time')
            print(f"Interpolated missing datetime values in '{column}'")
    
    return database

Database_After_Imputation = Imputation_Of_Values(MarketCampsdt)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())



CopiedMarkeCampsdt = Database_After_Imputation.copy()
print(" Dataset Loaded and Cleaned")
print("Shape:", CopiedMarkeCampsdt.shape)
print("Columns:", CopiedMarkeCampsdt.columns.tolist())


print("I am trying to make the cluster of the following coloums of the dataset here")


CopiedMarkeCampsdt["TotalSpend"] = (CopiedMarkeCampsdt["MntWines"] + CopiedMarkeCampsdt["MntFruits"] + CopiedMarkeCampsdt["MntMeatProducts"] +
                    CopiedMarkeCampsdt["MntFishProducts"] + CopiedMarkeCampsdt["MntSweetProducts"] + CopiedMarkeCampsdt["MntGoldProds"])

CopiedMarkeCampsdt["Children"] = CopiedMarkeCampsdt["Kidhome"] + CopiedMarkeCampsdt["Teenhome"]
CopiedMarkeCampsdt["Age"] = 2025 - CopiedMarkeCampsdt["Year_Birth"] 


CopiedMarkeCampsdt["Dt_Customer"] = pd.to_datetime(CopiedMarkeCampsdt["Dt_Customer"], format='%d-%m-%Y')



print("I am Selecting the Features That Needs to Be Taken for the Clustering Here")
features = [
    "Income", "Recency", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth", "TotalSpend", "Children", "Age"
]

FinalFeatSets = CopiedMarkeCampsdt[features]


print("I am Doing the Standardization Step that Needs to Be executed , that I am doing here")
scaler = StandardScaler()
FinalFeatSets_scaled = scaler.fit_transform(FinalFeatSets)


ClusterSum_Sqt = []
silhouette_scores = []

Rangedt = range(2, 11)
for k in Rangedt:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(FinalFeatSets_scaled)
    ClusterSum_Sqt.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(FinalFeatSets_scaled, kmeans.labels_))


print("I am Executing the Elbow Plot asked in the Pdf Here!")
plt.figure(figsize=(10,5))
plt.plot(Rangedt, ClusterSum_Sqt, 'bo-', linewidth=2)
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within Cluster Sum of Squares (WCSS)")
plt.show()


print("I am Executing the Silhouetter Plot asked in the Pdf Here !")
plt.figure(figsize=(10,5))
plt.plot(Rangedt, silhouette_scores, 'ro-', linewidth=2)
plt.title("Silhouette Score for Different k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

optimal_k = 4


print("I am Directly Apploying the K means Clustering Here after Performing the Above  Following Steps!")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
CopiedMarkeCampsdt["Cluster"] = kmeans.fit_predict(FinalFeatSets_scaled)

print("Clustering completed with", optimal_k, "clusters.")
print(CopiedMarkeCampsdt["Cluster"].value_counts())


the_comp_pca = PCA(n_components=2)
X_the_comp_pca = the_comp_pca.fit_transform(FinalFeatSets_scaled)

plt.figure(figsize=(10,7))
sns.scatterplot(x=X_the_comp_pca[:,0], y=X_the_comp_pca[:,1], hue=CopiedMarkeCampsdt["Cluster"], palette="Set2", s=60)
plt.title("Customer Clusters (PCA 2D Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()


brifclust = CopiedMarkeCampsdt.groupby("Cluster")[features + ["TotalSpend"]].mean().round(2)
print("\n Cluster Summary (mean values):\n")
print(brifclust)

plt.figure(figsize=(10,6))
sns.heatmap(brifclust, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Cluster Profile Heatmap")
plt.show()


overall_avg_spend = CopiedMarkeCampsdt["TotalSpend"].mean()
overall_avg_income = CopiedMarkeCampsdt["Income"].mean()

print("\n Recommendations and Insights:")
for c in brifclust.index:
    print(f"\n--- Cluster {c} ---")
    avg_income = brifclust.loc[c, "Income"]
    if isinstance(avg_income, pd.Series):
       avg_income = avg_income.iloc[0]

    avg_spend = brifclust.loc[c, "TotalSpend"]
    if isinstance(avg_spend, pd.Series):
        avg_spend = avg_spend.iloc[0]

    avg_age = brifclust.loc[c, "Age"]
    if isinstance(avg_age, pd.Series):
        avg_age = avg_age.iloc[0]


    print(f"Average Income: {avg_income:.2f}, Average Spend: {avg_spend:.2f}, Average Age: {avg_age:.1f}")
    if avg_spend > overall_avg_spend and avg_income > overall_avg_income:
        print(" Premium Customers — target with loyalty & high-value offers.")
    elif avg_spend > overall_avg_spend and avg_income < overall_avg_income:
        print(" Value Seekers — offer bundle deals or discounts.")
    elif avg_spend < overall_avg_spend and avg_income > overall_avg_income:
        print(" High-income but low-engagement — focus on personalized marketing.")
    else:
        print(" Low spenders — nurture through awareness campaigns.")