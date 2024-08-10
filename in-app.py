import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open('medical_insurance_cost_predictor.sav', 'rb'))

def medical_insurance_cost_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title('Medical Insurance Prediction Web App')
    
    try:
        age = float(st.text_input('Age'))
        sex = float(st.text_input('Sex: 0 -> Female, 1 -> Male'))
        bmi = float(st.text_input('Body Mass Index'))
        children = float(st.text_input('Number of Children'))
        smoker = float(st.text_input('Smoker: 0 -> No, 1 -> Yes'))
        region = float(st.text_input('Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest'))

        if st.button('Predict Medical Insurance Cost'):
            diagnosis = medical_insurance_cost_prediction([age, sex, bmi, children, smoker, region])
            st.write(f"Predicted Medical Insurance Cost: ${diagnosis:.2f}")

    except ValueError:
        st.error('Please enter valid numeric values.')

if __name__ == '__main__':
    main()