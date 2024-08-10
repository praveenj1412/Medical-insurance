import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv(r"D:\data\insurance.csv")
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['region'] = encoder.fit_transform(df['region'])
df['smoker'] = encoder.fit_transform(df['smoker'])

X = df.drop(columns='charges', axis=1)
Y = df['charges']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, Y_train)
testing_data_prediction = model.predict(X_test)
score = metrics.r2_score(Y_test, testing_data_prediction)
print(f"Model R^2 Score: {score}")

# Save the model
filename = 'medical_insurance_cost_predictor.sav'
pickle.dump(model, open(filename, 'wb'))

