📘 Student Score Prediction — Machine Learning Project

Predicting a student’s academic performance based on multiple factors using a full end-to-end Machine Learning pipeline.

This project uses a real-world dataset containing features related to a student’s background, study patterns, and academic behavior to predict their final exam score.
It includes data ingestion, exploratory data analysis, preprocessing, model selection, training, evaluation, and deployment.

📂 Project Structure
├── data
│   └── stud.csv
├── src
│   ├── components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── utils.py
├── artifacts
├── notebook
│   └── EDA.ipynb
├── app.py (Flask/FastAPI)
├── Dockerfile
├── requirements.txt
└── README.md

🧠 Problem Statement

The goal of this project is to build a machine learning model that predicts a student's final score using the following features:

📌 Columns Used

Gender

Race/Ethnicity

Parental Level of Education

Lunch Type

Test Preparation Course

Math Score

Reading Score

Writing Score

These features are processed and used to predict the overall performance score of the student.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Matplotlib & Seaborn

Flask / FastAPI (for deployment)

Docker (for containerization)

Railway / Render / Cloud Run (for cloud deployment)

🚀 Features of the Project
✔ End-to-End ML Pipeline

Includes ingestion → transformation → training → evaluation → prediction.

✔ Robust Preprocessing

Handling missing values

One-Hot Encoding

Standard Scaling for numeric columns

ColumnTransformer pipeline

✔ Trained Multiple Algorithms

Evaluated:

Linear Regression

Lasso

Ridge

KNN Regressor

Decision Tree

Random Forest

XGBoost (optional)

Final model chosen based on best R² score.

✔ Model Deployment Ready

Predict pipeline for real-time inference

Flask/FastAPI API endpoint

Dockerfile for deployment

Cloud deployment supported (Railway/Render)
