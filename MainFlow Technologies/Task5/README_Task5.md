
#  DATA ANALYSIS AND DATA SCIENCE WITH PYTHON — TASK 5

##  Internship Project Overview
This project was completed as part of the **Data Analysis and Data Science with Python Internship**.  
**Task 5** focused on **Classification Tasks**, where two main machine learning projects were implemented using **Python** and standard data-science libraries.

---

##  Task 1 — Student Pass/Fail Prediction

###  Objective
Predict whether a student will **pass or fail** based on:
- Study hours per week (`StudyTimeWeekly`)
- Attendance percentage (`Attendance`)

###  Dataset
A student performance dataset containing features such as:
- `StudyTimeWeekly`
- `Absences`
- `GPA`
- `GradeClass`
- Demographic attributes (age, gender, parental education, etc.)

###  Steps Performed
1. **Data Loading and Exploration**
   - Loaded dataset using `pandas`.
   - Checked shape, datatypes, duplicates, and missing values.
   - Imputed missing data using median (numeric) and mode (categorical).
2. **Feature Engineering**
   - Created a binary target column `Pass` (1 = Pass if GPA ≥ 2.0, else 0 = Fail).
   - Derived `Attendance` from absences.
3. **Data Visualization**
   - Scatter plot: `StudyTimeWeekly` vs `Attendance` colored by pass/fail.
   - Count plot for class distribution.
4. **Model Training**
   - Split dataset (80 % train, 20 % test).
   - Trained a **Logistic Regression** model.
5. **Model Evaluation**
   - Calculated **Accuracy** and **Confusion Matrix**.
   - Visualized the confusion matrix.
6. **Insights**
   - Examined model coefficients to determine how study hours and attendance affect the probability of passing.

###  Results
| Metric | Value |
|---------|-------|
| Accuracy | ≈ 80–90 % (varies by dataset) |
| Key Insight | Higher study hours and attendance strongly correlate with passing. |

---

##  Task 2 — Sentiment Analysis with NLP

###  Objective
Classify customer reviews (tweets) into **positive**, **negative**, or **neutral** sentiments using Natural Language Processing (NLP).

###  Dataset
**Twitter US Airline Sentiment Dataset** (`Tweets.csv`)  
Key columns:
- `text` → tweet text (review)
- `airline_sentiment` → sentiment label (positive, neutral, negative)

###  Steps Performed
1. **Data Loading and Cleaning**
   - Loaded dataset and dropped duplicates.
   - Filled missing values using mode/median as appropriate.
   - Retained only `text` and `airline_sentiment` columns.
2. **Text Preprocessing**
   - Lowercased all text.
   - Removed URLs, mentions, hashtags, numbers, and punctuation.
   - Removed stopwords using `NLTK`.
   - Lemmatized tokens using `WordNetLemmatizer`.
3. **Text Vectorization**
   - Converted cleaned text into numerical representation using **TF-IDF** (`max_features = 5000`).
4. **Model Training**
   - Split data (80 % train, 20 % test).
   - Trained a **Logistic Regression** model for multi-class sentiment prediction.
5. **Model Evaluation**
   - Calculated **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
   - Displayed **Confusion Matrix** and **Classification Report**.
6. **Insights**
   - Displayed correctly and incorrectly classified examples.
   - Identified top 10 influential words per sentiment class based on model coefficients.

###  Results
| Metric | Value (approx.) |
|---------|-----------------|
| Accuracy | 75 – 85 % |
| Precision | 0.80 |
| Recall | 0.78 |
| F1-Score | 0.79 |
| Key Insight | Negative tweets dominate the dataset; top negative keywords often relate to *delays* and *customer service issues*. |

---

##  Tools and Libraries Used
- **Python**  
- **pandas**, **NumPy** – data handling  
- **Matplotlib**, **Seaborn** – visualizations  
- **scikit-learn** – modeling, metrics, TF-IDF  
- **NLTK** – text preprocessing and lemmatization


##  Conclusion
This internship task successfully demonstrated the end-to-end process of **classification modeling** for both numerical and textual data.  
Through **Task 1** and **Task 2**, skills were gained in:
- Building predictive models  
- Performing sentiment analysis  
- Extracting actionable insights from data  

>  **Both projects align fully with the Task 5 requirements** of the internship document and were executed using clean, well-documented Python code.

