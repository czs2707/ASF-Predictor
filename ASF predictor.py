import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('best_model.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

# Define feature names from the new dataset
feature_names = [
    "CRRTTherapy", "Sodium", "Lactate", "PPI", "VasoactiveDrugs", "CRT", "OIIndex", "APACHEII", "SMSSpots", "PCT", "MV"
]

# Streamlit user interface
st.title("ICU elderly Sepsis patients ASF Predictor")

# Sodium: numerical input
Sodium = st.number_input("Sodium:", min_value=0, max_value=200, value=135)

# CRRTTherapy: categorical selection
CRRTTherapy = st.selectbox("CRRTTherapy:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# VasoactiveDrugs: categorical selection
VasoactiveDrugs = st.selectbox("VasoactiveDrugs:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Lactate: numerical input
Lactate = st.number_input("Lactate:", min_value=0.001, max_value=50, value=0.01)

# PPI: numerical input
PPI = st.number_input("PPI:", min_value=0.001, max_value=50, value=0.50)

# CRT: numerical input
CRT = st.number_input("CRT:", min_value=0.1, max_value=20, value=1)

# OIIndex: numerical input
OIIndex = st.number_input("OIIndex:", min_value=0.1, max_value=1000.0, value=300.0)

# APACHEII: numerical input
APACHEII = st.number_input("APACHEII:", min_value=1, max_value=100, value=20)

# SMSSpots: numerical input
SMSSpots = st.number_input("SMSSpots:", min_value=1, max_value=5, value=1)

# PCT: numerical input
PCT = st.number_input("PCT:", min_value=0.001, max_value=500.001, value=0.01)

# MV: categorical selection
MV = st.selectbox("MV:", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Process inputs and make predictions
feature_values = [CRRTTherapy, Sodium, Lactate, PPI, VasoactiveDrugs, CRT, OIIndex, APACHEII, SMSSpots, PCT, MV]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: ASF, 0: No ASF)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of ASF. "
            f"The model predicts that your probability of having ASF is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of ASF. "
            f"The model predicts that your probability of not having ASF is {probability:.1f}%. "
            "However, Regular Skin nursing is important. Please continue regular check-ups with your healthcare provider."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # Display the SHAP force plot for the predicted class
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['No ASF', 'ASF'],  # Adjust class names to match your classification task
        mode='classification'
    )

    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)  # Disable feature value table
    st.components.v1.html(lime_html, height=800, scrolling=True)