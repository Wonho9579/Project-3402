
![Logo](./project_images/UTA_logo.PNG)

# Loan Approval Prediction - Binary Classification Project

**One Sentence Summary:**  
This repository presents a binary classification model to predict loan approval status using a synthetic dataset based on financial application features.  
(Kaggle-style structure, data link: https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)

---

## Overview

This project aims to predict whether a loan will be approved based on four key features: CreditScore, AnnualIncome, EmploymentStatus, and TotalDebtToIncomeRatio.  
We used a Logistic Regression model to evaluate prediction performance.  
The final model achieved an **accuracy of 86.8%** and an **F1-score of 0.70** for the approved class, despite class imbalance.

---

## Summary of Workdone

### Data

- **Type**: Tabular, structured CSV file  
- **Input features**:  
  - CreditScore (numerical)  
  - AnnualIncome (numerical)  
  - EmploymentStatus (categorical: Employed, Self-Employed, Unemployed)  
  - TotalDebtToIncomeRatio (numerical)  
- **Target variable**: LoanApproved (binary: 0 = Not Approved, 1 = Approved)  
- **Size**: 20,000 samples  
- **Train/Test split**: 80% train, 20% test  

No missing values were present. No validation set was used, as model complexity was low.

**Initial Data Overview:**

![Step 2 Preview](./project_images/step2_data_preview_1.PNG)
![Step 2 Preview](./project_images/step2_data_preview.PNG)

> These previews confirm the structure and cleanliness of the dataset.  
> The first image shows a snapshot of the top 5 rows to help others understand the input format.  
> The second image shows column data types, counts, and summary statistics. Together, they justify skipping missing value handling and confirm readiness for preprocessing.

### Preprocessing / Clean up

- One-hot encoding applied to EmploymentStatus  
- StandardScaler applied to the 3 numerical features  
- Outliers visualized but not removed  
- Features were scaled to ensure comparability during training

**Feature Summary Table**

![Feature Summary Table](./project_images/feature_summary_table.PNG)

> This table guided our preprocessing decisions. Features with long ranges or many outliers (like AnnualIncome, DTI) were scaled.  
> EmploymentStatus, being categorical, was transformed using one-hot encoding. The absence of missing values justified no imputation.

### Data Visualization

**Credit Score by Loan Approval**

![CreditScore](./project_images/credit_score_by_approval.png)

> CreditScore shows clear class separation. Most approved applicants scored over 600, whereas rejections concentrated below 580.  
> This visualization confirms CreditScore's predictive value and supports keeping it in the model.

**Annual Income by Loan Approval**

![AnnualIncome](./project_images/annual_income_by_approval.png)

> Although income is skewed, the 40K–80K range dominates among approved cases. This plot illustrates income’s secondary but relevant role.  
> High-income outliers exist but do not dominate prediction.

**Debt-to-Income Ratio by Loan Approval**

![DTI](./project_images/dti_by_approval.png)

> This plot reveals that applicants with DTI > 1 are rarely approved, confirming it as a high-risk signal.  
> Low DTI strongly correlates with approvals, making this a valuable feature.

**Employment Status Distribution**

![EmploymentStatus](./project_images/employment_bar.png)

> The bar chart shows most approvals came from Employed applicants. Self-Employed and Unemployed statuses correlate with rejection.  
> This supports the categorical feature’s strong contribution to classification.

---

### Problem Formulation

- **Input**: `[CreditScore, AnnualIncome, EmploymentStatus_onehot, TotalDebtToIncomeRatio]`  
- **Output**: `LoanApproved (0 or 1)`  
- **Model Used**: LogisticRegression  
- **Why?**: Simple, interpretable, fast to train. Ideal for binary classification on tabular data.

---

### Training

- **Environment**: Google Colab  
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn  
- **Training Time**: Less than 1 minute  
- **No tuning or early stopping used** due to model simplicity  
- **No issues** encountered during training

**Scaling Results**

> The following before/after scaling plots visually confirm that scaling was applied correctly and helps ensure fair feature contribution.  
> This is especially important since Logistic Regression is sensitive to feature magnitudes.

**Credit Score: Before vs After Scaling**

![Scaled CreditScore](./project_images/scaled_credit_score.png)

> The bell-shaped curve is preserved, but the center is normalized around 0.  
> This eliminates scale bias and allows fair weighting during training.

**Annual Income: Before vs After Scaling**

![Scaled Income](./project_images/scaled_annual_income.png)

> Originally right-skewed with extreme values, AnnualIncome is now in a comparable range.  
> Though skew remains, scaling reduces the model’s risk of being dominated by a few large values.

**Debt-to-Income Ratio: Before vs After Scaling**

![Scaled DTI](./project_images/scaled_dti.png)

> DTI was heavily skewed with extreme outliers. After scaling, these outliers are controlled.  
> This helps stabilize the training process and ensures numerical stability for the model.

---

### Performance Comparison

![Classification Report](./project_images/classification_report.PNG)

- **Accuracy**: 86.8%  
- **Precision (class 1)**: 0.77  
- **Recall (class 1)**: 0.63  
- **F1-score (class 1)**: 0.70  

> This confusion matrix and classification report show the model performs well on majority and minority classes.  
> Precision > recall suggests conservative approval prediction, which aligns with real-world risk policies.

---

## Conclusions

- CreditScore and DebtRatio are the most influential features  
- EmploymentStatus shows clear class separation  
- Logistic Regression was a suitable and stable choice for baseline prediction

---

## Future Work

- Try tree-based models: RandomForest, GradientBoosting  
- Use validation split or cross-validation  
- Apply SMOTE to rebalance the dataset  
- Add feature importance or SHAP plots for model interpretability

---

## How to Reproduce Results

1. Open `Loan_Project_Wonho_Jeong.ipynb` in Colab or Jupyter  
2. Run all cells sequentially  
3. Outputs include preprocessing, visualizations, model performance, and a Kaggle-style submission file

---

## Overview of Files in Repository

| File | Description |
|------|-------------|
| `Loan_Project_Wonho_Jeong.ipynb` | Main code notebook |
| `loan_approval_submission_Wonho Jeong.csv` | Final test predictions |
| `README.md` | Project documentation |
| `project_images/` | All analysis and result visualizations |
| `README_TEMPLATE.md` | Original structure from professor |

---

## Software Setup / Data

```bash
pip install pandas scikit-learn matplotlib seaborn
```

- Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval)

---

## Citations

- Zoppelletto, Lorenzo. *Financial Risk for Loan Approval*, Kaggle.  
- scikit-learn documentation  
- Logistic Regression literature
