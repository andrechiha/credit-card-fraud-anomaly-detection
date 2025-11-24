Credit Card Fraud Anomaly Detection

This project implements an unsupervised machine learning pipeline to detect fraudulent credit card transactions using the well-known Kaggle dataset.
The goal is to identify anomalies without using any labeled training.

Overview

The final anomaly detection system is built using:

StandardScaler for feature scaling

PCA (25 components) for dimensionality reduction

Isolation Forest as the anomaly detection model

All preprocessing objects (scaler, PCA) and the best model are saved as .pkl files for deployment.

Dataset

Source: Kaggle – Credit Card Fraud Detection
Rows: 284,807
Fraud Rate: ~0.172%
Target column:

0 → Normal transaction

1 → Fraudulent transaction

Features include anonymized PCA components V1–V28, plus Amount and Time.

Final Model

After testing K-Means, Local Outlier Factor (LOF), One-Class SVM, and Isolation Forest, the best results were obtained with Isolation Forest.

Best Parameters:

n_estimators = 100
max_samples = 'auto'
contamination = 0.0016
max_features = 1.0

Performance on the full dataset:
Metric	Fraud (Class 1)
Precision	0.33
Recall	0.31
F1-score	0.32

This performance is strong for an unsupervised model on a highly imbalanced dataset.

Repository Contents
File	Description
unsupervised_projectipynb.ipynb	Full notebook (EDA, PCA, tuning, evaluation)
scaler.pkl	Fitted StandardScaler
pca_transformer.pkl	Fitted PCA transformer
isolation_forest_model.pkl	Best Isolation Forest model
model_metadata.json	Model parameters and metadata
How to Use
Install dependencies
pip install numpy pandas scikit-learn

Load the saved model and preprocessing
import pickle
import numpy as np

# Load preprocessing
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca_transformer.pkl", "rb") as f:
    pca = pickle.load(f)

# Load model
with open("isolation_forest_model.pkl", "rb") as f:
    iso_forest = pickle.load(f)

Make predictions
def predict_fraud(X):
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    pred = iso_forest.predict(X_pca)
    return np.where(pred == -1, 1, 0)   # -1 = fraud

Models Tested

K-Means

Local Outlier Factor (LOF)

One-Class SVM

Isolation Forest (best)

Future Work

Testing Autoencoders for anomaly detection

Trying supervised models with SMOTE

Deploying API using FastAPI

Adding a dashboard for visualization

Author

andrechiha
