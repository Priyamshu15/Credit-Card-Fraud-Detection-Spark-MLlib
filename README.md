# Credit Card Fraud Detection using Apache Spark MLlib

An end-to-end machine learning project that detects fraudulent credit card transactions using Apache Spark and PySpark MLlib.

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem because fraudulent transactions represent a very small proportion of total transactions.

This project uses distributed data processing and machine learning algorithms to identify potentially fraudulent transactions and compare model performance.

## Dataset

- Total transactions: 284,807
- Fraudulent transactions: Approximately 0.17%
- Problem type: Binary Classification
- Train-test split: 80:20

Note: The dataset is not included in this repository because of its large file size.

## Technologies Used

- Python
- Apache Spark
- PySpark MLlib
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Machine Learning Workflow

1. Data loading and exploration
2. Data preprocessing
3. Handling class imbalance using class weights
4. Feature transformation using VectorAssembler
5. Feature scaling using StandardScaler
6. Model training using Logistic Regression and Random Forest
7. Model evaluation and comparison

## Models Implemented

- Logistic Regression
- Random Forest Classifier

## Model Performance

| Evaluation Metric | Random Forest |
|---|---:|
| Precision | 0.75 |
| Recall | 0.84 |
| F1-Score | 0.79 |
| AUC-PR | 0.79 |

Random Forest achieved better performance than Logistic Regression, particularly in identifying fraudulent transactions.

## Visualizations

The `charts/` directory contains:

- Class distribution
- Model comparison
- Confusion matrices
- Effect of partitions on training time

## Project Structure

Credit-Card-Fraud-Detection-Spark/
│
├── charts/
│   ├── 1_class_distribution.png
│   ├── 2_model_comparison.png
│   ├── 3_confusion_matrices.png
│   └── 4_partition_time.png
│
├── fraud_detection.py
├── README.md
├── requirements.txt
└── .gitignore

## How to Run

1. Install Python and Apache Spark.
2. Install the required Python libraries.
3. Place the dataset file `creditcard.csv` in the project directory.
4. Run the Python script:

```bash
python fraud_detection.py