import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("bagging_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
fare = st.number_input("Fare", min_value=0.0, step=0.1)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"])

# Encoding inputs
sex = 1 if sex == "Male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_map[embarked]

# Prepare input
features = np.array([[pclass, sex, age, fare, embarked]])

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.success("🎉 The passenger is likely to SURVIVE.")
    else:
        st.error("💀 The passenger is likely to NOT survive.")
