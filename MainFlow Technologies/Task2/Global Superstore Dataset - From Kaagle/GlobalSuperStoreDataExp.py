
# Importing The Necessary Librarires for the Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Loading the Dataset using the read_csv function
storedealincv = pd.read_csv('superstore.csv')

print(storedealincv.describe())

#Providing a short summary of all the coloums in the dataset
print(storedealincv.info())
print("\n")
print("\n\n");
# Displaying the first 5 rows of the dataset    
print("First 5 rows of the dataset:")

print(storedealincv.head().to_string())

#Identify and remove exact/partial duplicate rows
Duplicatestored = storedealincv.duplicated().sum()
print("Number of duplicated rows: ",Duplicatestored )
print("The Duplicated Rows are: \n", storedealincv[storedealincv.duplicated()])
print("\n")
datacleannotduplicate = storedealincv.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", datacleannotduplicate.head().to_string())
print("\n")


PartialCopyRowSum = storedealincv.duplicated(subset= storedealincv.columns.difference(['Index']) , keep="first").sum()
print("The No People Database Partial Duplicates Rows are: ", PartialCopyRowSum)
print("\n")

StoreNodupecol = storedealincv.drop_duplicates(subset = storedealincv.columns.difference(['Index']) , keep = 'first')
print(StoreNodupecol.head().to_string())


print("\n Since This is Not an Exact Part of the Intenship Task We will not be going further with this . I wil continue with the Duplicates removed dataframe itself here !\n")

crossstorescol = ['Category','City','Country','Customer.ID','Customer.Name','Discount',
 'Market','记录数','Order.Date','Order.ID','Order.Priority','Product.ID',
 'Product.Name','Profit','Quantity','Region','Row.ID','Sales','Segment',
 'Ship.Date','Ship.Mode','Shipping.Cost','State','Sub.Category','Year',
 'Market2','weeknum']

paritalDuplicateRows = []

for index, row in StoreNodupecol.iterrows():
    for i in range(len(crossstorescol)):
        for j in range(i+1, len(crossstorescol)):
            datacal1 = str(row[crossstorescol[i]]).lower()
            datacal2 = str(row[crossstorescol[j]]).lower()
            if datacal1 and datacal1 != "nan" and datacal1 in datacal2:
                paritalDuplicateRows.append((index, crossstorescol[i], crossstorescol[j], row[crossstorescol[i]], row[crossstorescol[j]]))

print("Rows with cross-column redundancy:")
for r in paritalDuplicateRows:
    print(r)


print("Missing Values in each Column: \n", StoreNodupecol.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (StoreNodupecol.isnull().mean()*100))

misscolsofstores = pd.DataFrame({
    'Missing_Values_Coloums': StoreNodupecol.isnull().any(),
    'Missing_Values_Count': StoreNodupecol.isnull().sum(),
    'Missing_ValuesIn_Percentage': (StoreNodupecol.isnull().sum() / len(StoreNodupecol)) * 100
})

print(misscolsofstores.to_string())

def MissedValueFiller(database):
    for column in database.columns:
        if np.issubdtype(database[column].dtype, np.number):
            median_value = database[column].median()
            database[column].fillna(median_value)
            print(f"Filled NaN in numeric column '{column}' with Median = {median_value}")
        
        elif database[column].dtype == 'object':
            mode_value = database[column].mode()[0]
            database[column].fillna(mode_value)
            print(f"Filled NaN in categorical column '{column}' with Mode = {mode_value}")
        
        elif np.issubdtype(database[column].dtype, np.datetime64):
            database[column] = database[column].interpolate(method='time')
            print(f"Interpolated missing datetime values in '{column}'")
    
    return database
filledDatasetstore = MissedValueFiller(StoreNodupecol)
print("The Database After Imputation of Missing Values is : \n", filledDatasetstore.head().to_string())

class StoreDataAnalysis:
    def __init__(self, dataframe):
        self.copiedDataset = dataframe.copy()   

    def OutlierAnlays(self, column, threshold=3):
        z = np.abs(stats.zscore(self.copiedDataset[column]))
        before = len(self.copiedDataset)
        self.copiedDataset = self.copiedDataset[z < threshold]
        after = len(self.copiedDataset)
        print(f"Removed {before - after} outliers from '{column}' using z-score")
        return self.copiedDataset

    def StaticAnalys(self, colstobeanalyzed):
        print("\nSummary:\n", self.copiedDataset[colstobeanalyzed].describe())
        print("\nMean:\n", self.copiedDataset[colstobeanalyzed].mean())
        print("\nMedian:\n", self.copiedDataset[colstobeanalyzed].median())
        print("\nMode:\n", self.copiedDataset[colstobeanalyzed].mode().iloc[0])
        print("\nStd Dev:\n", self.copiedDataset[colstobeanalyzed].std())
        print("\nVariance:\n", self.copiedDataset[colstobeanalyzed].var())
        print("\nCorrelations:\n", self.copiedDataset[colstobeanalyzed].corr())

    def HistogrmaAnalys(self, colstobeanalyzed):
        self.copiedDataset[colstobeanalyzed].hist(bins=20, figsize=(12,8))
        plt.suptitle("Histograms")
        plt.show()

    def BoxplotAnalys(self, colstobeanalyzed):
        plt.figure(figsize=(10,5))
        sns.boxplot(data=self.copiedDataset[colstobeanalyzed])
        plt.title("Boxplots")
        plt.show()

    def HeatMapAnalys(self, colstobeanalyzed):
        plt.figure(figsize=(8,6))
        sns.heatmap(self.copiedDataset[colstobeanalyzed].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()


colstobeanalyzed = ["Sales","Profit","Discount","Quantity","Shipping.Cost"]
analysis = StoreDataAnalysis(StoreNodupecol)
for col in colstobeanalyzed:
    analysis.OutlierAnlays(col)
analysis.StaticAnalys(colstobeanalyzed)
analysis.HistogrmaAnalys(colstobeanalyzed)
analysis.BoxplotAnalys(colstobeanalyzed)
analysis.HeatMapAnalys(colstobeanalyzed)

