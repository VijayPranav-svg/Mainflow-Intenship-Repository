import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

MallCstdt = pd.read_csv('Mall_Customers.csv')
MallCstdt.columns = MallCstdt.columns.str.strip()

# Basic inspection
print(MallCstdt.head().to_string())
print("\n")
print(MallCstdt.info())
print("\n")
print(MallCstdt.shape)
print("\n")
print("Data Shape:", MallCstdt.shape)
print("\nData Types:\n", MallCstdt.dtypes)
print("\nMissing Values:\n", MallCstdt.isnull().sum())
print("\nSummary Statistics Of Data:\n", MallCstdt.describe(include='all'))

DuplicateRowsum = MallCstdt.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", MallCstdt[MallCstdt.duplicated()])
print("\n")
database_withDuplicatedRowsGone = MallCstdt.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")

Partial_Duplicates_Rows_Count = MallCstdt.duplicated(keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
MallCstdt_no_partial_duplicates = MallCstdt.drop_duplicates(keep='first')
print(MallCstdt_no_partial_duplicates.head().to_string())

print("Missing Values in each Column: \n", MallCstdt.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (MallCstdt.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': MallCstdt.isnull().any(),
    'Missing_Values_Count': MallCstdt.isnull().sum(),
    'Missing_ValuesIn_Percentage': (MallCstdt.isnull().sum() / len(MallCstdt)) * 100
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

Database_After_Imputation = Imputation_Of_Values(MallCstdt)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())


print("\n As a first step i am loading the copy of the inital cleaned dataset that i have created! \n ")
CopyMallCsdt = Database_After_Imputation.copy()
print("Shape:", CopyMallCsdt.shape)
print(CopyMallCsdt.head().to_string())


print("I am trying to make the cluster of the following coloums of the dataset here")
if 'Genre' in CopyMallCsdt.columns:
    genre_mode = CopyMallCsdt['Genre'].mode()[0]
    CopyMallCsdt['Genre'] = CopyMallCsdt['Genre'].fillna(genre_mode)
    CopyMallCsdt['Genre_Encoded'] = CopyMallCsdt['Genre'].map({'Male': 0, 'Female': 1})
    if CopyMallCsdt['Genre_Encoded'].isnull().any():
        CopyMallCsdt['Genre_Encoded'] = CopyMallCsdt['Genre'].astype('category').cat.codes
else:
    CopyMallCsdt['Genre_Encoded'] = 0


print("I am Selecting the Features That Needs to Be Taken for the Clustering Here")

features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)',
    'Genre_Encoded'
]


features = [f for f in features if f in CopyMallCsdt.columns]
CopyMallCsdt_features = CopyMallCsdt[features].astype(float)


print("I am Doing the Standardization Step that Needs to Be executed , that I am doing here")
scaler = StandardScaler()
CopyMallCsdt_scaled = scaler.fit_transform(CopyMallCsdt_features)


MallCustMeanHld = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(CopyMallCsdt_scaled)
    MallCustMeanHld.append(km.inertia_)
    try:
        intermedsilscr = silhouette_score(CopyMallCsdt_scaled, labels)
    except Exception:
        intermedsilscr = np.nan
    silhouette_scores.append(intermedsilscr)

print("I am Executing the Elbow Plot asked in the Pdf Here!")
plt.figure(figsize=(9,4))
plt.plot(k_values, MallCustMeanHld, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_values)
plt.show()

print("I am Executing the Silhouetter Plot asked in the Pdf Here !")
plt.figure(figsize=(9,4))
plt.plot(k_values, silhouette_scores, marker='o', color='tab:orange')
plt.title('Silhouette Score by k — higher is better (closer to 1)')
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.xticks(k_values)
plt.show()


print("I am Directly Apploying the K means Clustering Here after Performing the Above  Following Steps!")

kmostrangd = int(k_values[np.nanargmax(silhouette_scores)])
print(f"\nSelected number of clusters (by silhouette argmax): k = {kmostrangd}")

kmeans = KMeans(n_clusters=kmostrangd, random_state=42, n_init=10)
CopyMallCsdt['Cluster'] = kmeans.fit_predict(CopyMallCsdt_scaled)

print("\nCluster counts:\n", CopyMallCsdt['Cluster'].value_counts().sort_index())



pltcltcentrod = PCA(n_components=2, random_state=42)
X_pltcltcentrod = pltcltcentrod.fit_transform(CopyMallCsdt_scaled)

plt.figure(figsize=(8,6))
palette = sns.color_palette("tab10", kmostrangd)
sns.scatterplot(x=X_pltcltcentrod[:,0], y=X_pltcltcentrod[:,1], hue=CopyMallCsdt['Cluster'], palette=palette, s=60, alpha=0.9, edgecolor='k')
plt.title(f'K-Means clusters (k={kmostrangd}) — PCA 2D projection')
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 2')
plt.legend(title='Cluster')
plt.show()


centroids_scaled = kmeans.cluster_centers_ 
centroids_pltcltcentrod = pltcltcentrod.transform(centroids_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pltcltcentrod[:,0], y=X_pltcltcentrod[:,1], hue=CopyMallCsdt['Cluster'], palette=palette, s=40, alpha=0.6, legend=False)
plt.scatter(centroids_pltcltcentrod[:,0], centroids_pltcltcentrod[:,1], marker='X', s=200, c='black', label='Centroids')
for i, (x,y) in enumerate(centroids_pltcltcentrod):
    plt.text(x, y, f'  C{i}', fontsize=12, weight='bold')
plt.title('Clusters with Centroids (PCA space)')
plt.xlabel('PCA 1'); plt.ylabel('PCA 2')
plt.legend()
plt.show()


Devpltclsts = CopyMallCsdt[features + ['Cluster']].copy()
Devpltclsts['Cluster'] = Devpltclsts['Cluster'].astype('category')
sns.pairplot(Devpltclsts, hue='Cluster', diag_kind='kde', corner=False, palette=palette)
plt.suptitle("Pairplot of features colored by cluster", y=1.02)
plt.show()

cluster_profile = CopyMallCsdt.groupby('Cluster')[features].agg(['mean','count']).round(2)
cluster_means = CopyMallCsdt.groupby('Cluster')[features].mean().round(2)
cluster_sizes = CopyMallCsdt['Cluster'].value_counts().sort_index()

print("\nCluster sizes:\n", cluster_sizes)
print("\nCluster means:\n", cluster_means)

plt.figure(figsize=(8,4))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Cluster profile (mean values)")
plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.show()



overall_income = CopyMallCsdt['Annual Income (k$)'].mean()
overall_spend = CopyMallCsdt['Spending Score (1-100)'].mean()

print("\nRecommendations and short insights by cluster:")
for c in sorted(CopyMallCsdt['Cluster'].unique()):
    row = cluster_means.loc[c]
    size = int(cluster_sizes.loc[c])
    age_mean = float(row['Age'])
    income_mean = float(row['Annual Income (k$)'])
    spend_mean = float(row['Spending Score (1-100)'])
    print(f"\n--- Cluster {c} (n={size}) ---")
    print(f"Avg Age: {age_mean:.1f}, Avg Income (k$): {income_mean:.1f}, Avg SpendingScore: {spend_mean:.1f}")
    # simple rules:
    if spend_mean > overall_spend and income_mean > overall_income:
        print("-> Premium & engaged: target with premium upsell, loyalty programs.")
    elif spend_mean > overall_spend and income_mean <= overall_income:
        print("-> High spend, lower income: promote value bundles and financing options.")
    elif spend_mean <= overall_spend and income_mean > overall_income:
        print("-> High-income but low spend: personalized outreach, exclusive offers to boost engagement.")
    else:
        print("-> Low spenders: awareness campaigns, email nurture and discount experiments.")

