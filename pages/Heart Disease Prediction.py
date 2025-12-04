import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide"
)

# Title and description
st.title("❤️ Heart Failure Prediction System")
st.markdown("""
This application predicts the likelihood of **heart failure** based on various health parameters.
Fill in the form below to get your prediction.
""")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/logr.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'logr.pkl' not found in models/ directory!")
        return None

# Load individual scalers
@st.cache_resource
def load_scalers():
    scalers = {}
    try:
        with open('scalers/scaler_age.pkl', 'rb') as f:
            scalers['Age'] = pickle.load(f)
        with open('scalers/scaler_cholesterol.pkl', 'rb') as f:
            scalers['Cholesterol'] = pickle.load(f)
        with open('scalers/scaler_maxhr.pkl', 'rb') as f:
            scalers['MaxHR'] = pickle.load(f)
        with open('scalers/scaler_oldpeak.pkl', 'rb') as f:
            scalers['Oldpeak'] = pickle.load(f)
    except FileNotFoundError:
        st.error("One or more scaler files not found in scalers/ directory!")
    return scalers

# Load resources
model = load_model()
scalers = load_scalers()

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
    chest_pain = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
        format_func=lambda x: {0:'ASY - Asymptomatic',1:'ATA - Atypical Angina',2:'NAP - Non-Anginal Pain',3:'TA - Typical Angina'}[x])
    cholesterol = st.number_input("Cholesterol (mm/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col2:
    st.subheader("Medical Parameters")
    
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", options=[0, 1, 2],
        format_func=lambda x: {0:'Down - Downsloping',1:'Flat - Flat',2:'Up - Upsloping'}[x])

# Prediction button
st.markdown("---")
if st.button("🔍 Predict Heart Failure Risk", type="primary", use_container_width=True):

    if model is None or not scalers:
        st.error("Cannot make prediction! Model or scalers not loaded.")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })

        # Apply each individual scaler
        for feature, scaler in scalers.items():
            if feature in input_data.columns:
                input_data[feature] = scaler.transform(input_data[[feature]])

        # Select features for prediction
        features_to_use = ['Age', 'Sex', 'ChestPainType', 'Cholesterol', 'FastingBS',
                           'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
        input_features = input_data[features_to_use].values

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Try to get probabilities
        try:
            probability = model.predict_proba(input_features)[0]
            prob_no_failure = probability[0] * 100
            prob_failure = probability[1] * 100
        except:
            prob_no_failure = prob_failure = None

        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        if prediction == 1:
            st.error("⚠️ **HIGH RISK**: The model predicts a high likelihood of **heart failure**.")
        else:
            st.success("✅ **LOW RISK**: The model predicts a low likelihood of **heart failure**.")

        if prob_failure is not None:
            st.markdown("### Prediction Confidence")
            c1, c2 = st.columns(2)
            c1.metric("No Heart Failure", f"{prob_no_failure:.1f}%")
            c2.metric("Heart Failure", f"{prob_failure:.1f}%")
            st.progress(prob_failure / 100)

        # Input summary
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                'Parameter': ['Age','Sex','Chest Pain Type','Cholesterol','Fasting BS','Max HR','Exercise Angina','Oldpeak','ST Slope'],
                'Value': [age, 'Male' if sex==1 else 'Female', chest_pain, cholesterol,
                          'Yes' if fasting_bs==1 else 'No', max_hr, 
                          'Yes' if exercise_angina==1 else 'No', oldpeak, st_slope]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses **Logistic Regression** for **heart failure** risk prediction.
    
    ### Features Used
    - Age, Sex, Chest Pain Type, Cholesterol, FastingBS, MaxHR, Exercise Angina, Oldpeak, ST Slope
    
    ### Disclaimer
    This tool is for **educational purposes only**. Always consult a healthcare professional.
    """)
