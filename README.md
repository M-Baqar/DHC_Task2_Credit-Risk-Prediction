# Task 2: Credit Risk Prediction — Loan Default

> **Internship:** Data Science & Analytics — DevelopersHub Corporation  
> **Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Google Colab

---

## Task Objective

Financial institutions face significant losses when loan applicants fail to repay their loans. The objective of this task is to build a **binary classification model** that predicts whether a loan applicant is likely to **default** (not repay) based on their personal and financial information.

This helps banks make faster, data-driven loan approval decisions and reduce the risk of financial loss.

**Target Variable:** `Loan_Status` — Y (Approved / No Default Risk) or N (Rejected / Default Risk)

---

## Approach

### 1. Data Loading
- Loaded the Loan Prediction Dataset from a public CSV mirror (auto-downloads in the notebook)
- Manual upload option also provided for Google Colab users via Kaggle download

### 2. Data Cleaning & Handling Missing Values
- Identified all columns with missing values and visualized them using a bar chart
- **Categorical columns** (Gender, Married, Dependents, Self_Employed, Credit_History, Loan_Amount_Term): filled missing values with the **mode** (most frequent value)
- **Numerical column** (LoanAmount): filled missing values with the **median** to avoid distortion from outliers
- Confirmed zero missing values after cleaning

### 3. Exploratory Data Analysis (EDA)
Visualizations created to understand the data:

| Chart | Insight Sought |
|-------|---------------|
| Bar chart — Loan Status distribution | Class balance check |
| Histogram — Loan Amount | Distribution shape and skewness |
| Count plot — Education vs Loan Status | Does education affect approval? |
| Box plot — Income vs Loan Status | Income differences between approved/rejected |
| Scatter plot — Income vs Loan Amount | Relationship between income and requested amount |
| Count plot — Credit History vs Loan Status | Impact of credit history on approval |

### 4. Feature Encoding
- Dropped the `Loan_ID` column (identifier, not a feature)
- Applied **Label Encoding** to all remaining categorical columns to convert them to numeric format

### 5. Model Training
- Split data: **80% training / 20% testing** with stratified sampling to preserve class balance
- Trained two models:
  - **Logistic Regression** (max_iter=1000)
  - **Decision Tree Classifier** (max_depth=5)

### 6. Model Evaluation
- Evaluated both models using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)
- Plotted side-by-side confusion matrices and accuracy comparison bar chart

---

## Results & Insights

### Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~80–82% |
| Decision Tree | ~76–80% |

Logistic Regression performed better and is preferred due to its stability and interpretability.

### Key Findings

1. **Credit History is the most important factor** — applicants with good credit history (value = 1) have a dramatically higher loan approval rate. This single feature has the strongest influence on the outcome.

2. **Income alone is not enough** — some high-income applicants still get rejected, primarily due to poor credit history or very high requested loan amounts relative to their income.

3. **Education improves approval chances** — graduates have a higher approval rate than non-graduates, though credit history still dominates.

4. **Loan Amount distribution is right-skewed** — most applicants request moderate loan amounts, but a few request very large amounts which can affect approval.

5. **Missing Data Pattern** — Credit History had the most missing values (~8%). Since it is the most predictive feature, careful imputation was important.

### Business Impact
The model can automate initial loan screening, significantly reducing manual processing time. By flagging high-risk applicants early, banks can reduce default rates while approving creditworthy customers faster.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle — Loan Prediction Problem Dataset |
| Kaggle Link | https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset |
| Rows | 614 |
| Columns | 13 |
| Target | `Loan_Status` (Y / N) |
| Missing Values | Yes — handled via mode/median imputation |
