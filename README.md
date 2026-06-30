## Credit Card Fraud Detection System URL for Web page

https://credit-card-fraud-detection-system-n5jbftxkytezkcmd2jykds.streamlit.app/

# 💳 Credit Card Fraud Detection System

A Machine Learning project that detects fraudulent credit card transactions using classification algorithms and addresses the challenge of highly imbalanced data through **SMOTE (Synthetic Minority Oversampling Technique)**.

---

## 📌 Project Overview

Credit card fraud detection is a critical problem in the financial industry due to the increasing number of online transactions. Since fraudulent transactions represent only a tiny fraction of all transactions, traditional machine learning models often become biased toward legitimate transactions.

This project builds and compares machine learning models to identify fraudulent transactions while improving detection performance by handling class imbalance using SMOTE.

---

## 🎯 Problem Statement

The objective of this project is to develop a machine learning model capable of accurately detecting fraudulent credit card transactions from historical transaction data. The primary challenge is the severe class imbalance, where fraudulent transactions account for less than 1% of the dataset.

---

## 🚀 Features

- Data preprocessing and exploration
- Binary classification for fraud detection
- Logistic Regression implementation
- Random Forest implementation
- Handling class imbalance using SMOTE
- Model performance comparison before and after oversampling

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Imbalanced-learn (SMOTE)
- **Development Environment:** Jupyter Notebook

---

## 📂 Project Workflow

1. Load and inspect the dataset.
2. Perform data preprocessing.
3. Split data into training and testing sets.
4. Train baseline models:
   - Logistic Regression
   - Random Forest
5. Apply SMOTE to balance the minority class.
6. Retrain the Random Forest model.
7. Compare model performance before and after oversampling.

---

## 🤖 Machine Learning Models

### Logistic Regression
- Used as the baseline classification model.
- Simple, fast, and interpretable.
- Performance is limited on highly imbalanced datasets.

### Random Forest
- Ensemble learning algorithm based on multiple decision trees.
- Better captures complex patterns in transaction data.
- Demonstrates improved fraud detection after balancing the dataset.

---

## 📊 Results

- Successfully identified the impact of class imbalance on fraud detection.
- Improved fraud detection performance using SMOTE.
- Random Forest trained on the balanced dataset outperformed the baseline models in identifying fraudulent transactions.

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit Card Fraud Detection.ipynb
├── creditcard.csv
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

---

## 🔮 Future Improvements

- Apply SMOTE only to the training dataset to avoid data leakage.
- Perform hyperparameter tuning.
- Evaluate using Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- Deploy the model using Flask or FastAPI.
- Explore advanced algorithms such as XGBoost and LightGBM.

---

## 📚 Key Learnings

- Working with highly imbalanced datasets.
- Oversampling techniques using SMOTE.
- Building classification models with Scikit-learn.
- Comparing machine learning algorithms.
- Understanding fraud detection challenges in real-world financial systems.

---

## 👨‍💻 Author

**Devendra Kumar**

B.Tech Graduate | Data Analytics | Machine Learning | SQL | Python | Power BI


