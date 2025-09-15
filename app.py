
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and the scaler
model = joblib.load('Credit_Card_Model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and description
st.title('Credit Card Fraud Detection System')
st.write('Enter the values for the transaction features to get a prediction.')

# Create input fields
input_features = {}
for i in range(1, 29):
    key = f'v{i}'
    input_features[key] = st.number_input(f'Enter value for {key}', value=0.0)

amount = st.number_input('Enter value for Amount', value=0.0)

if st.button('Predict'):
    # Prepare the input data
    v_features = [input_features[f'v{i}'] for i in range(1, 29)]
    scaled_amount = scaler.transform([[amount]])[0][0]
    final_features_list = v_features + [scaled_amount]
    final_features = np.array(final_features_list).reshape(1, -1)
    
    # Make the prediction
    y_pred = model.predict(final_features)[0]

    # Display the result
    st.write('---')
    if y_pred == 0:
        st.success('Prediction: Normal Transaction (Class 0)')
    else:
        st.error('Prediction: Fraudulent Transaction (Class 1)')