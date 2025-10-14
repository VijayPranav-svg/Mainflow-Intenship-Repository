import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


AmesHstData = pd.read_csv('AmesHousing.csv')
print(AmesHstData.head())
print("\n")
print(AmesHstData.info())
print("\n")
print(AmesHstData.shape)
print("\n")
print("Data Shape:", AmesHstData.shape)
print("\nData Types:\n", AmesHstData.dtypes)
print("\nMissing Values:\n", AmesHstData.isnull().sum())
print("\nSummary Statistics Of Data:\n", AmesHstData.describe(include='all'))

DuplicateRowsum = AmesHstData.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", AmesHstData[AmesHstData.duplicated()])
print("\n")
database_withDuplicatedRowsGone = AmesHstData.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head())
print("\n")


Partial_Duplicates_Rows_Count = AmesHstData.duplicated(keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
AmesHstData_no_partial_duplicates = AmesHstData.drop_duplicates(keep='first')
print(AmesHstData_no_partial_duplicates.head())


print("Missing Values in each Column: \n", AmesHstData.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (AmesHstData.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': AmesHstData.isnull().any(),
    'Missing_Values_Count': AmesHstData.isnull().sum(),
    'Missing_ValuesIn_Percentage': (AmesHstData.isnull().sum() / len(AmesHstData)) * 100
})

print(Missing_Values_Coloums_Dataframe)


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

Database_After_Imputation = Imputation_Of_Values(AmesHstData)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head())


print("\nDescriptive Statistics:\n", Database_After_Imputation.describe())
plt.figure(figsize=(6,4))
sns.histplot(Database_After_Imputation["SalePrice"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.show()



print("\nVisualizing Outliers with Boxplots for Key Numeric Features...\n")
ColsReferds = Database_After_Imputation.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15,10))
for i, col in enumerate(["Lot Area", "Gr Liv Area", "SalePrice", "Total Bsmt SF", "Garage Area", "1st Flr SF"]):
    if col in Database_After_Imputation.columns:
        plt.subplot(2,3,i+1)
        sns.boxplot(x=Database_After_Imputation[col])
        plt.title(f"{col}")
plt.tight_layout()
plt.show()



X = Database_After_Imputation.drop("SalePrice", axis=1)
y = Database_After_Imputation["SalePrice"]


X = X.drop(["Order", "PID"], axis=1, errors="ignore")


NumResps = X.select_dtypes(include=["int64", "float64"]).columns
StrResps = X.select_dtypes(include=["object"]).columns

print(f"\nNumeric Columns: {len(NumResps)}")
print(f"Categorical Columns: {len(StrResps)}")




NumResps_Trans = StandardScaler()
StrResps_Trans = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', NumResps_Trans, NumResps),
        ('cat', StrResps_Trans, StrResps)
    ])


plt.figure(figsize=(10,8))
corr = Database_After_Imputation.select_dtypes(include=['number']).corr()
sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False),
            annot=True, cmap='coolwarm', cbar=False)
plt.title("Correlation of Numerical Features with SalePrice")
plt.show()
plt.figure(figsize=(8,12))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTrain shape:", X_train.shape, " | Test shape:", X_test.shape)

AmesMdls = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
AmesMdls.fit(X_train, y_train)


Predsts = AmesMdls.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, Predsts))
r2 = r2_score(y_test, Predsts)

print("\nModel Evaluation Results:")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=Predsts)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

Numfeatures = NumResps  
Strfeatures = StrResps  

encoded_cat_features = list(
    AmesMdls.named_steps['preprocessor']
    .transformers_[1][1]
    .get_feature_names_out(Strfeatures)
)


EncdFelist = list(Numfeatures) + encoded_cat_features

Coeffs = AmesMdls.named_steps['regressor'].coef_

FetsDatfs = pd.DataFrame({
    'Feature': EncdFelist,
    'Coefficient': Coeffs
}).sort_values(by='Coefficient', ascending=False)

print("\nTop Positive Predictors:")
print(FetsDatfs.head(10))

print("\nTop Negative Predictors:")
print(FetsDatfs.tail(10))
