# Task 5 - Consumer Complaint Text Classification

## Overview
This project performs **text classification** on the Consumer Complaint dataset from CFPB.  
The goal is to classify consumer complaints into the following categories:

| Label | Category |
|-------|---------|
| 0     | Credit reporting, repair, or other |
| 1     | Debt collection |
| 2     | Consumer Loan |
| 3     | Mortgage |

We use **TF-IDF vectorization** and **Logistic Regression** as the baseline model.  
Advanced versions use **Sentence-BERT embeddings** or **DistilBERT fine-tuning** for higher accuracy.

---

## Dataset
- Source: [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)  
- Columns used:
  - `Product` → mapped to target labels
  - `Consumer complaint narrative` → main text input
  - `Issue` and `Sub-issue` → used if narrative is empty

---

## Steps Followed

1. **Data Loading & Cleaning**
    - Removed empty narratives and combined with `Product`, `Issue`, `Sub-issue`.
    - Cleaned text: lowercase, removed punctuation, normalized whitespace.

2. **Feature Engineering**
    - TF-IDF vectorization (uni-grams, bi-grams, max_features=20000)
    - Optionally, Sentence-BERT embeddings for improved performance

3. **Model Selection**
    - Baseline: Logistic Regression
    - Alternatives tested: Naive Bayes, SVM
    - Advanced: XGBoost with SBERT embeddings, DistilBERT fine-tuning

4. **Model Training**
    - Split dataset: 80% train, 20% test
    - Trained classifier on TF-IDF vectors
    - Evaluated with accuracy, precision, recall, F1-score

5. **Model Evaluation**
    - Confusion Matrix
    - Multi-class ROC-AUC curves

6. **Prediction**
    - Sample predictions on new complaint texts

---

## Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
nltk
joblib
sentence-transformers  # optional for SBERT
xgboost                # optional for advanced model
transformers           # optional for fine-tuned DistilBERT
datasets               # optional for fine-tuned DistilBERT


#Install with
pip install pandas numpy scikit-learn matplotlib nltk joblib sentence-transformers xgboost transformers datasets


Usage:
import joblib
import pandas as pd

# Load trained model
final_model = joblib.load("models/final_model.joblib")

# Prepare TF-IDF vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

# Make prediction
sample = ["I am facing issues with my mortgage payments."]
sample_vec = tfidf.transform(sample)
pred_label = final_model.predict(sample_vec)[0]

# Map numeric label to category
label_map = {
    0: "Credit reporting, repair, or other",
    1: "Debt collection",
    2: "Consumer Loan",
    3: "Mortgage"
}
print(f"The complaint is classified as: {label_map[pred_label]}")
