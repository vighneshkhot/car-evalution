#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# car_app.py
pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('car.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    le = LabelEncoder()
    df = df.apply(le.fit_transform)
    return df, le

# Train model
def train_model(df):
    X = df.drop('class', axis=1)
    y = df['class']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# UI
st.title("🚗 Car Evaluation Classifier")
df, le = load_data()
model = train_model(df)

# Input
buying = st.selectbox("Buying Price", df.columns[:-1], index=0)
maint = st.selectbox("Maintenance", df.columns[:-1], index=1)
doors = st.selectbox("Doors", df.columns[:-1], index=2)
persons = st.selectbox("Persons", df.columns[:-1], index=3)
lug_boot = st.selectbox("Lug Boot Size", df.columns[:-1], index=4)
safety = st.selectbox("Safety", df.columns[:-1], index=5)

# Predict
if st.button("Classify"):
    sample = [[
        le.transform([buying])[0],
        le.transform([maint])[0],
        le.transform([doors])[0],
        le.transform([persons])[0],
        le.transform([lug_boot])[0],
        le.transform([safety])[0]
    ]]
    pred = model.predict(sample)[0]
    label_map = {v: k for k, v in le.classes_.items()}
    st.success(f"Predicted Class: **{label_map[pred]}**")

