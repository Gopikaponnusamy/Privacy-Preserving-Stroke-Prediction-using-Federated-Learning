# Privacy-Preserving Stroke Prediction using Federated Learning

## Overview

This project implements a privacy-preserving machine learning system for predicting the risk of stroke using a healthcare dataset. It demonstrates how Federated Learning (FL) principles can be applied to train models collaboratively without sharing raw data. The system compares centralized machine learning with a simulated federated learning approach and visualizes the results through an interactive dashboard.

---

## Objectives

* Predict stroke risk using medical data
* Preserve patient privacy using federated learning principles
* Compare centralized learning with federated learning performance
* Build an interactive dashboard for visualization

---

## Project Structure

```text
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
```

---

## Tech Stack

### Machine Learning

* Scikit-learn — model building and evaluation (Logistic Regression)
* NumPy — numerical computations
* Pandas — data handling and preprocessing

### Federated Learning (Simulation)

* Custom federated learning simulation using multiple client datasets
* Distributed data partitioning
* Aggregation of client model performance

### Data Processing

* Label Encoding for categorical variables
* Missing value handling (BMI imputation)
* Train-test split for evaluation

### Visualization

* Matplotlib for plotting accuracy comparisons
* Streamlit for interactive dashboard visualization

### Frontend

* Streamlit for building the user interface

  * Model comparison panel
  * Accuracy charts
  * Privacy insights
  * Federated learning workflow

### Backend

* Python for core logic
* Modular scripts:

  * preprocess.py for data cleaning
  * centralized.py for centralized training
  * fl_simulation.py for federated simulation
  * app.py for dashboard

### Data Storage

* CSV for dataset storage
* JSON for storing model results

### Development Tools

* VS Code
* Git and GitHub
* Python virtual environment (venv)

---

## Dataset

The project uses a healthcare stroke dataset with the following features:

* Gender, Age
* Hypertension, Heart Disease
* Marital Status
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status
* Stroke (target variable)

---

## System Workflow

1. Load and preprocess the dataset
2. Train a centralized machine learning model
3. Simulate federated learning across multiple clients
4. Aggregate model results
5. Compare performance
6. Visualize results in the dashboard

---

## Model Comparison

| Model Type  | Description                        |
| ----------- | ---------------------------------- |
| Centralized | Trained on complete dataset        |
| Federated   | Trained on distributed client data |

---

## Privacy Benefits

* No raw data sharing between clients
* Decentralized learning approach
* Reduced risk of data leakage
* Suitable for sensitive domains like healthcare

---

## Risks and Solutions

### Risks

* Model inversion attacks
* Gradient leakage
* Non-IID data distribution

### Solutions

* Differential privacy techniques
* Secure aggregation methods
* Robust and regularized training

---

## How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Run centralized model

```
python centralized.py
```

### Run federated simulation

```
python fl_simulation.py
```

### Launch dashboard

```
streamlit run app.py
```

---

## Output

* Centralized model accuracy
* Federated model accuracy
* Accuracy comparison charts
* Interactive dashboard

---

## Applications

* Healthcare data collaboration
* Remote medical diagnosis systems
* Privacy-preserving AI systems
* Medical research without data sharing

---

## Key Highlights

* Privacy-preserving machine learning implementation
* Federated learning concept demonstration
* Healthcare-based AI application
* Interactive visualization dashboard
* Clean and modular project structure

---

## Future Enhancements

* Integration with real federated learning frameworks such as Flower
* Cloud deployment (Azure or AWS)
* Advanced evaluation metrics such as ROC curve and confusion matrix
* Real-time multi-client training

---

## Author

Gopika Ponnusamy
