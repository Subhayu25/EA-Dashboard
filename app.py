import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("EA.csv")
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Employees")
departments = st.sidebar.multiselect("Select Department", options=df['Department'].unique(), default=df['Department'].unique())
job_roles = st.sidebar.multiselect("Select Job Role", options=df['JobRole'].unique(), default=df['JobRole'].unique())
genders = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())

filtered_df = df[(df['Department'].isin(departments)) & 
                 (df['JobRole'].isin(job_roles)) & 
                 (df['Gender'].isin(genders))]

# Title
st.title("📊 HR Analytics Dashboard – Employee Attrition")
st.markdown("This dashboard helps the HR Director and leadership team analyze key drivers of employee attrition.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Demographics", "Tenure & Salary", "Performance", "Predictive Insights"])

# --------------------------------------------
with tab1:
    st.header("🔍 Executive Summary")
    st.markdown("Understand the overall employee count, attrition rate, average age and tenure.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", len(filtered_df))
    col2.metric("Attrition Rate", f"{100*filtered_df['Attrition'].mean():.2f}%")
    col3.metric("Average Age", f"{filtered_df['Age'].mean():.1f} years")
    col4.metric("Avg Tenure (Years)", f"{filtered_df['YearsAtCompany'].mean():.1f}")

    # Chart 1
    st.subheader("Attrition by Department")
    fig = px.bar(filtered_df, x='Department', color='Attrition', barmode='group', title="Attrition Count by Department")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 2
    st.subheader("Attrition by Education Field")
    fig = px.histogram(filtered_df, x='EducationField', color='Attrition', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
with tab2:
    st.header("🧑‍🤝‍🧑 Demographic Analysis")

    # Chart 3
    st.subheader("Gender Distribution")
    fig = px.pie(filtered_df, names='Gender', title="Gender Proportion")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 4
    st.subheader("Age vs Attrition")
    fig = px.box(filtered_df, x='Attrition', y='Age', points="all", title="Age Distribution by Attrition")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 5
    st.subheader("Marital Status and Attrition")
    fig = px.histogram(filtered_df, x='MaritalStatus', color='Attrition', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
with tab3:
    st.header("💰 Tenure and Compensation")

    # Chart 6
    st.subheader("Monthly Income Distribution")
    fig = px.histogram(filtered_df, x='MonthlyIncome', nbins=30, title="Monthly Income Histogram")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 7
    st.subheader("Tenure vs Monthly Income")
    fig = px.scatter(filtered_df, x='YearsAtCompany', y='MonthlyIncome', color='Attrition', title="Income vs Tenure")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 8
    st.subheader("Years at Company vs Attrition")
    fig = px.box(filtered_df, x='Attrition', y='YearsAtCompany', points="all", title="Tenure Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Chart 9
    st.subheader("Job Level vs Monthly Income")
    fig = px.box(filtered_df, x='JobLevel', y='MonthlyIncome', color='Attrition')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
with tab4:
    st.header("📈 Performance & Work Factors")

    # Chart 10
    st.subheader("Job Satisfaction by Attrition")
    fig = px.histogram(filtered_df, x='JobSatisfaction', color='Attrition', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # Chart 11
    st.subheader("Work Life Balance by Attrition")
    fig = px.histogram(filtered_df, x='WorkLifeBalance', color='Attrition', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # Chart 12
    st.subheader("OverTime and Attrition")
    fig = px.histogram(filtered_df, x='OverTime', color='Attrition', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # Chart 13
    st.subheader("Distance From Home vs Attrition")
    fig = px.box(filtered_df, x='Attrition', y='DistanceFromHome')
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
with tab5:
    st.header("🤖 Predictive Insights (Feature Importance)")
    st.markdown("Based on Random Forest Classifier to understand top factors driving attrition.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Preprocessing
    df_ml = df.copy()
    df_ml = pd.get_dummies(df_ml, drop_first=True)
    X = df_ml.drop("Attrition", axis=1)
    y = df_ml["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(15)

    # Chart 14
    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Top 15 Features Driving Attrition")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
st.markdown("---")
st.caption("Created by Subhayu | Powered by Streamlit | Data Source: EA.csv")
