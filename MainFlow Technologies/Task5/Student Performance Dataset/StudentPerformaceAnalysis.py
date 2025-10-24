import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay




print("\n==============================")
print(" TASK 1: STUDENT PASS/FAIL PREDICTION")
print("==============================")


print("\n I am Loading the dataset into a dataframe...")

SutudentData = pd.read_csv('Student_performance_data _.csv')
print(SutudentData.head().to_string())
print("\n")
print(SutudentData.info())
print("\n")
print(SutudentData.shape)
print("\n")
print("Data Shape:", SutudentData.shape)
print("\nData Types:\n", SutudentData.dtypes)
print("\nMissing Values:\n", SutudentData.isnull().sum())
print("\nSummary Statistics Of Data:\n", SutudentData.describe(include='all'))

print("\n I am Exploring the datasets structure and basic information...")
print("\n Information about columns and datatypes and etc: \n")

DuplicateRowsum = SutudentData.duplicated().sum()
print("Number of duplicated rows: ",DuplicateRowsum )
print("The Duplicated Rows are: \n", SutudentData[SutudentData.duplicated()])
print("\n")
database_withDuplicatedRowsGone = SutudentData.drop_duplicates()
print("The Database Without the Duplicaed Rows Here \n ", database_withDuplicatedRowsGone.head().to_string())
print("\n")

Partial_Duplicates_Rows_Count = SutudentData.duplicated(keep=False).sum()
print("The No of Partial Duplicates Rows are: ", Partial_Duplicates_Rows_Count)
print("\n")
SutudentData_no_partial_duplicates = SutudentData.drop_duplicates(keep='first')
print(SutudentData_no_partial_duplicates.head().to_string())



print("Missing Values in each Column: \n", SutudentData.isnull().sum())
print("\n")
print("Percentage of Missing Values in each Column: \n", (SutudentData.isnull().mean()*100))

Missing_Values_Coloums_Dataframe = pd.DataFrame({
    'Missing_Values_Coloums': SutudentData.isnull().any(),
    'Missing_Values_Count': SutudentData.isnull().sum(),
    'Missing_ValuesIn_Percentage': (SutudentData.isnull().sum() / len(SutudentData)) * 100
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

Database_After_Imputation = Imputation_Of_Values(SutudentData)
print("The Database After Imputation of Missing Values is : \n", Database_After_Imputation.head().to_string())


print("\n\n Statistical Explanation Defined Here : \n\n")
print(Database_After_Imputation.describe())


# -----------------------------------------------
print("\n I am Now Creating a binary target column 'Pass' based on GPA values...")
Database_After_Imputation['Pass'] = np.where(Database_After_Imputation['GPA'] >= 2.0, 1, 0)
print("New column 'Pass' created successfully (1 = Pass, 0 = Fail).")

print("\n I have Calculating attendance percentage from absences assuming 30 total classes...")
Database_After_Imputation['Attendance'] = 100 - (Database_After_Imputation['Absences'] / 30 * 100)
Database_After_Imputation['Attendance'] = Database_After_Imputation['Attendance'].clip(lower=0)
print("Attendance column added successfully!\n")

print(Database_After_Imputation[['StudyTimeWeekly', 'Absences', 'Attendance', 'GPA', 'Pass']].head())



print("\n Now we are Visualizing relationships between study time, attendance, and pass/fail outcomes...")

plt.figure(figsize=(6,4))
sns.scatterplot(x='StudyTimeWeekly', y='Attendance', hue='Pass', data=Database_After_Imputation, palette='coolwarm')
plt.title('Study Time vs Attendance (Colored by Pass/Fail)')
plt.show()

plt.figure(figsize=(5,3))
sns.countplot(x='Pass', data=Database_After_Imputation, palette='Set2')
plt.title('Pass/Fail Distribution')
plt.show()

print("\nVisualizations displayed successfully!")


print("\n Selecting important features for model training...")
X = Database_After_Imputation[['StudyTimeWeekly', 'Attendance']]
y = Database_After_Imputation['Pass']
print("Features selected: StudyTimeWeekly and Attendance")
print("Target variable: Pass")

print("\nSplitting dataset into training and testing sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


print("\n Training Logistic Regression model on training data...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")


print("\n At Last I am  Evaluating model performance on the test set...")
y_pred = model.predict(X_test)

ModelAccury = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n Model Accuracy: {ModelAccury*100:.2f}%")
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


print("\n Analyzing which features have the most impact on prediction...")
coef_Database_After_Imputation = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

print("\nFeature Importance (Model Coefficients):\n")
print(coef_Database_After_Imputation)

plt.figure(figsize=(5,3))
sns.barplot(x='Feature', y='Coefficient', data=coef_Database_After_Imputation, palette='viridis')
plt.title("Feature Impact on Passing Probability")
plt.show()

print("\nPositive coefficients indicate features that increase the likelihood of passing,")
print("while negative coefficients indicate features that decrease it.")
print("\n Task 1 completed successfully!")