import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from centralized import train_centralized, train_centralized_model
from fl_simulation import federated_training
from evaluation_plots import plot_confusion_matrix, plot_learning_curve

st.set_page_config(page_title="FL Stroke Prediction", layout="wide")

st.markdown("""
<style>
.stApp {background:#0e1117;}
* {color:white !important;}
input {background:white !important;color:black !important;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Federated Learning - Stroke Prediction")

page = st.radio("Navigation", ["Dashboard", "Prediction"])

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    if st.button("Run Training"):

        central_acc = train_centralized()
        fl_history, fl_metrics = federated_training()

        st.subheader("Comparison")

        col1, col2 = st.columns(2)
        col1.metric("Centralized", f"{central_acc:.4f}")
        col2.metric("Federated", f"{fl_metrics['accuracy']:.4f}")

        st.subheader("Metrics")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Precision", f"{fl_metrics['precision']:.4f}")
        c2.metric("Recall", f"{fl_metrics['recall']:.4f}")
        c3.metric("F1", f"{fl_metrics['f1']:.4f}")
        c4.metric("ROC-AUC", f"{fl_metrics['roc_auc']:.4f}")

        st.pyplot(plot_confusion_matrix(fl_metrics["conf_matrix"]))
        st.pyplot(plot_learning_curve(fl_history, fl_history))

# ---------------- PREDICTION ----------------
else:

    model = train_centralized_model()

    age = st.number_input("Age")
    hyper = st.selectbox("Hypertension", [0,1])
    heart = st.selectbox("Heart Disease", [0,1])
    glucose = st.number_input("Glucose")
    bmi = st.number_input("BMI")
    gender = st.selectbox("Gender", [0,1])
    married = st.selectbox("Married", [0,1])
    work = st.selectbox("Work Type", [0,1,2,3,4])

    if st.button("Predict"):

        X = np.array([[age, hyper, heart, glucose, bmi, gender, married, work]], dtype=np.float64)

        pred = model.predict(X)[0]

        if pred == 1:
            st.error("High Stroke Risk ⚠")
        else:
            st.success("Low Stroke Risk ✅")