import numpy as np
import pandas as pd
import streamlit as st
import joblib

loaded_model = joblib.load('medical_insurance_cost_predictor.sav')

def medical_insurance_cost_prediction(input_data):
    feature_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    input_data_df = pd.DataFrame([input_data], columns=feature_names)

    prediction = loaded_model.predict(input_data_df)
    return prediction

def main():
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0
    region_mapping = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region = region_mapping[region]

    if st.button("Predict"):
        result = medical_insurance_cost_prediction([age, sex, bmi, children, smoker, region])
        st.success(f"The predicted insurance cost is: ${result[0]:.2f}")

if __name__ == "__main__":
    main()
