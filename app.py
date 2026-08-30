import pandas as pd
import joblib
import streamlit as st

# --- Load model and encoders ---
model = joblib.load('credit_risk_model.pickle')

encoded_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account']
encoders = {col: joblib.load(f'{col}_encoder.pickle') for col in encoded_columns}
target_encoder = joblib.load('target_encoder.pickle')

# --- Page setup ---
st.title('Credit Risk Prediction App')
st.write('Enter applicant information to predict if the credit risk is good or bad.')

# --- Input fields ---
age = st.number_input('Age', min_value=18, max_value=80, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving accounts', ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox('Checking account', ['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit amount', min_value=0, value=1000)
duration = st.number_input('Duration (months)', min_value=1, value=12)

# --- Build model input ---
input_df = pd.DataFrame({
    'Age': [age],
    'Sex': encoders['Sex'].transform([sex]),
    'Job': [job],
    'Housing': encoders['Housing'].transform([housing]),
    'Saving accounts': encoders['Saving accounts'].transform([saving_accounts]),
    'Checking account': encoders['Checking account'].transform([checking_account]),
    'Credit amount': [credit_amount],
    'Duration': [duration],
})

# --- Prediction ---
if st.button('Predict Risk'):
    prediction = model.predict(input_df)[0]
    label = target_encoder.inverse_transform([prediction])[0]

    if label == 'good':
        st.success(f'The predicted credit risk is: **{label.upper()}**')
    else:
        st.error(f'The predicted credit risk is: **{label.upper()}**')
