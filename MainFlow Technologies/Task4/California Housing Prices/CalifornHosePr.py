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


Califorhosedt = pd.read_csv('housing.csv')
print(Califorhosedt.head().to_string())
print("\n")
print(Califorhosedt.info())
print("\n")
print(Califorhosedt.shape)
print("\n")
print("Data Shape:", Califorhosedt.shape)
print("\nData Types:\n", Califorhosedt.dtypes)
print("\nMissing Values:\n", Califorhosedt.isnull().sum())
print("\nSummary Statistics Of Data:\n", Califorhosedt.describe(include='all'))

DuplicateRowsum = Califorhosedt.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", Califorhosedt[Califorhosedt.duplicated()])
print("\n")
database_withDuplicatedRowsGone = Califorhosedt.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")

Partial_Duplicates_Rows_Count = Califorhosedt.duplicated(keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
Califorhosedt_no_partial_duplicates = Califorhosedt.drop_duplicates(keep='first')
print(Califorhosedt_no_partial_duplicates.head().to_string())

print("Missing Values in each Column: \n", Califorhosedt.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (Califorhosedt.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': Califorhosedt.isnull().any(),
    'Missing_Values_Count': Califorhosedt.isnull().sum(),
    'Missing_ValuesIn_Percentage': (Califorhosedt.isnull().sum() / len(Califorhosedt)) * 100
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

Database_After_Imputation = Imputation_Of_Values(Califorhosedt)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())


print("\n\n Statistical Explanation Defined Here : \n\n")
print(Database_After_Imputation.describe())

print("\n\n I am Showing the Distribution of the Target Variable Here ! \n\n")
plt.figure(figsize=(6,4))
sns.histplot(Database_After_Imputation['median_house_value'], bins=30, kde=True)
plt.title("Distribution of Median House Value")
plt.show()

print("\n\n Outlier Detection Using Boxplots for Key Numerical Features \n\n")
numeric_cols = Database_After_Imputation.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:6], 1):  
    plt.subplot(2, 3, i)
    sns.boxplot(x=Database_After_Imputation[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

print("\n\n I am Taking 2 Features and Seperating the Numerical and Categorical Coloumns from Each Other Here! \n\n")
X = Database_After_Imputation.drop('median_house_value', axis=1)
y = Database_After_Imputation['median_house_value']


NumCalifrna = X.select_dtypes(include=['int64', 'float64']).columns
CategCalifrna = ['ocean_proximity']


NumCalifrna_Trans = StandardScaler()
CategCalifrna_trans = OneHotEncoder(handle_unknown='ignore')

CaliProcesst = ColumnTransformer(
    transformers=[
        ('num', NumCalifrna_Trans, NumCalifrna),
        ('cat', CategCalifrna_trans, CategCalifrna)
    ])


print("\n\n I am Displaying the Correlation HeatMap for the Numeric Variables Here Clearly \n\n")

plt.figure(figsize=(10,8))
sns.heatmap(Database_After_Imputation.select_dtypes(include=['number']).corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()


print("\n \n I am Creating the Model and Its PipeLine Here \n\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


CaliforniaModel = Pipeline(steps=[
    ('CaliProcesst', CaliProcesst),
    ('regressor', LinearRegression())
])

CaliforniaModel.fit(X_train, y_train)

print("\n\n  I am doing the Prediction and Evaluation Metrics Here Respectively Here!")
y_pred = CaliforniaModel.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation: \n ")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

CaliComs = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPredicted vs Actual values: \n ")
print(CaliComs.head())


plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


CaliforniaEncFts = list(NumCalifrna) + list(CaliforniaModel.named_steps['CaliProcesst']
    .transformers_[1][1].get_feature_names_out(CategCalifrna))
coefs = CaliforniaModel.named_steps['regressor'].coef_

feature_importance = pd.DataFrame({
    'Feature': CaliforniaEncFts,
    'Coefficient': coefs
}).sort_values(by='Coefficient', ascending=False)

print("\nTop Predictors Influencing House Prices:")
print(feature_importance.head(10))