import streamlit as st
import joblib 
import pandas as pd

model=joblib.load("model/loan_model.pkl")
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦",layout="centered")

st.title("🏦 Loan Approval Prediction")
st.markdown("### ShadowFox Internship project")
st.write("enter the applicant details below to predict loan approval")
st.sidebar.title("📋 About")

st.sidebar.info("""
This application predicts whether a loan application
will be approved using Machine Learning.
Model Used:
✅ Logistic Regression
Accuracy:
86.18%
""")
st.markdown("-----")
st.subheader("Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income", min_value=0,value=5000)
co_income = st.number_input("Coapplicant Income", min_value=0,value=0)
loan_amount = st.number_input("Loan Amount", min_value=0,value=150)
loan_term = st.number_input("Loan Amount Term",min_value=12, value=360)
credit = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = {"0":0, "1":1, "2":2, "3+":3}[dependents]
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Rural":0, "Semiurban":1, "Urban":2}[property_area]

if st.button("🔍 Predict Loan Approval", use_container_width=True):
 input_data = pd.DataFrame([[gender, married, dependents, education,
                          self_employed, app_income, co_income,
                          loan_amount, loan_term, credit,
                          property_area]],
                        columns=[
                            "Gender","Married","Dependents","Education",
                            "Self_Employed","ApplicantIncome",
                            "CoapplicantIncome","LoanAmount",
                            "Loan_Amount_Term","Credit_History",
                            "Property_Area"
                        ])
prediction = model.predict(input_data)
probability=model.predict_proba(input_data)
st.markdown("----")
st.subheader("prediction Result")
if prediction[0] == 1:
        st.balloons()
        st.success("Congratulations 🎉Loan Approved")
        confidence=probability[0][1]
else:
        st.error("Sorry Loan Rejected 😥")
        confidence=probability[0][0]
st.write("### prediction Confidence ###")
st.progress(float(confidence))
st.write(f"Confidence: {confidence*100:.2f}%")        