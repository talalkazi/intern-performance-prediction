import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Intern Performance Prediction App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Intern Performance Prediction Model")
st.markdown("This application uses machine learning (**Random Forest Regressor**) to predict performance scores based on workplace and professional features. Adjust the parameters on the sidebar to test different scenarios!")

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('intern_performance.csv')
    X = df.drop(columns=['Employee_ID', 'Hire_Date', 'Performance_Score', 'Resigned'])
    y = df['Performance_Score']
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)
    rf.fit(X_train, y_train)
    return rf, X_encoded.columns, df

model, model_columns, raw_df = load_and_train_model()

st.sidebar.header("🔧 Input Intern Features")

department = st.sidebar.selectbox("Department", raw_df['Department'].unique())
job_title = st.sidebar.selectbox("Job Title", raw_df['Job_Title'].unique())
gender = st.sidebar.selectbox("Gender", raw_df['Gender'].unique())
education = st.sidebar.selectbox("Education Level", raw_df['Education_Level'].unique())

age = st.sidebar.slider("Age", int(raw_df['Age'].min()), int(raw_df['Age'].max()), 25)
years_at_company = st.sidebar.slider("Years At Company", int(raw_df['Years_At_Company'].min()), int(raw_df['Years_At_Company'].max()), 1)
monthly_salary = st.sidebar.slider("Monthly Salary ($)", float(raw_df['Monthly_Salary'].min()), float(raw_df['Monthly_Salary'].max()), 5000.0)
work_hours = st.sidebar.slider("Work Hours Per Week", int(raw_df['Work_Hours_Per_Week'].min()), int(raw_df['Work_Hours_Per_Week'].max()), 40)
projects = st.sidebar.slider("Projects Handled", int(raw_df['Projects_Handled'].min()), int(raw_df['Projects_Handled'].max()), 15)
overtime = st.sidebar.slider("Overtime Hours", int(raw_df['Overtime_Hours'].min()), int(raw_df['Overtime_Hours'].max()), 10)
sick_days = st.sidebar.slider("Sick Days", int(raw_df['Sick_Days'].min()), int(raw_df['Sick_Days'].max()), 3)
remote_freq = st.sidebar.slider("Remote Work Frequency (%)", int(raw_df['Remote_Work_Frequency'].min()), int(raw_df['Remote_Work_Frequency'].max()), 50)
team_size = st.sidebar.slider("Team Size", int(raw_df['Team_Size'].min()), int(raw_df['Team_Size'].max()), 10)
training_hours = st.sidebar.slider("Training Hours", int(raw_df['Training_Hours'].min()), int(raw_df['Training_Hours'].max()), 30)
promotions = st.sidebar.slider("Promotions", int(raw_df['Promotions'].min()), int(raw_df['Promotions'].max()), 0)
satisfaction = st.sidebar.slider("Employee Satisfaction Score", float(raw_df['Employee_Satisfaction_Score'].min()), float(raw_df['Employee_Satisfaction_Score'].max()), 3.5)

input_data = pd.DataFrame({
    'Age': [age],
    'Years_At_Company': [years_at_company],
    'Monthly_Salary': [monthly_salary],
    'Work_Hours_Per_Week': [work_hours],
    'Projects_Handled': [projects],
    'Overtime_Hours': [overtime],
    'Sick_Days': [sick_days],
    'Remote_Work_Frequency': [remote_freq],
    'Team_Size': [team_size],
    'Training_Hours': [training_hours],
    'Promotions': [promotions],
    'Employee_Satisfaction_Score': [satisfaction],
    'Department': [department],
    'Gender': [gender],
    'Job_Title': [job_title],
    'Education_Level': [education]
})

input_encoded = pd.get_dummies(input_data)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Prediction")
    if st.button("Predict Performance Score", type="primary"):
        prediction = model.predict(input_encoded)[0]
        st.success(f"### Predicted Performance Score: {prediction:.2f} / 5.0")
        
        if prediction >= 4.0:
            st.info("🌟 **Status:** Likely to Excel!")
        elif prediction >= 2.5:
            st.warning("⚠️️ **Status:** Average / Stable Performance.")
        else:
            st.error("🚨 **Status:** At Risk / Likely to Struggle.")

with col2:
    st.subheader("💡 Key Feature Importances")
    importances = pd.Series(model.feature_importances_, index=model_columns).sort_values(ascending=False).head(5)
    st.bar_chart(importances)

with st.expander("🔍 View Raw Dataset Sample"):
    st.dataframe(raw_df.head(10))