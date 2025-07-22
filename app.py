import streamlit as st
import joblib
import pandas as pd

st.title("🧠 Employee Salary Predictor")

exp = st.slider("Years of Experience", 0, 30, 5)
edu = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
role = st.selectbox("Role", ["Software Engineer", "Data Analyst", "HR Executive", "Manager"])
dept = st.selectbox("Department", ["IT", "Data", "HR", "Finance"])

if st.button("Predict Salary"):
    model = joblib.load("salary_model.pkl")
    input_df = pd.DataFrame([[exp, edu, role, dept]],
                            columns=["Experience", "Education Level", "Role", "Department"])
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Salary: ₹{int(prediction):,}")
