# Privacy-Preserving-Stroke-Prediction-using-Federated-Learning
Overview

This project implements a Privacy-Preserving Federated Learning (FL) system for predicting the risk of stroke using a healthcare dataset.

Instead of sharing raw patient data, multiple clients collaboratively train a machine learning model by sharing only model updates, ensuring data privacy and security.
Objectives
Predict stroke risk using medical data
Preserve patient privacy using Federated Learning principles
Compare Centralized Learning vs Federated Learning performance
Demonstrate real-world healthcare AI use case
Project Structure
stroke-fl/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── model.py
├── preprocess.py
├── centralized.py
├── fl_simulation.py
├── app.py
│
├── centralized_results.json
├── fl_results.json
└── requirements.txt
Tech Stack
*Machine Learning
Scikit-learn — Model building (Logistic Regression), training & evaluation
NumPy — Numerical computations and array handling
Pandas — Data loading, preprocessing, and manipulation
Federated Learning (Concept Simulation)
Custom FL Simulation — Simulated multi-client training and aggregation
Distributed Data Handling — Client-wise dataset partitioning
Model Aggregation Logic — Averaging client model performance
Data Processing & Preprocessing
Label Encoding — Converting categorical data to numerical format
Data Cleaning — Handling missing values (BMI imputation)
Train-Test Split — Model validation and evaluation
Data Visualization
Matplotlib — Accuracy comparison charts
Streamlit Charts — Interactive visualizations in dashboard
Frontend / Dashboard
Streamlit — Interactive web-based UI
Model comparison panel
Accuracy graph
Privacy insights
Federated workflow visualization
Backend Logic
Python — Core programming language
Modular scripts:
preprocess.py → Data preparation
centralized.py → Centralized training
fl_simulation.py → Federated simulation
app.py → Dashboard UI
TensorFlow (initial experimentation)
Flower Framework (FL framework exploration)
System Workflow
Load and preprocess dataset
Train Centralized ML model
Simulate Federated Learning across multiple clients
Aggregate results
Compare performance
Visualize results using Streamlit
