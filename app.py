import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Finance Analytics Dashboard", layout="wide")

# Load data
df = pd.read_csv("data/processed/features.csv")

df["date"] = pd.to_datetime(df["date"])
df["year_month"] = pd.to_datetime(df["year_month"])

st.title("💰 Finance Analytics Dashboard")

st.sidebar.header("Filters")

categories = sorted(df["super_category"].unique())

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    categories,
    default=categories
)

filtered = df[df["super_category"].isin(selected_categories)]

category_summary = (
    filtered.groupby("super_category")["abs_amount"]
    .sum()
    .reset_index()
    .sort_values("abs_amount", ascending=False)
)

fig = px.bar(
    category_summary,
    x="super_category",
    y="abs_amount",
    title="Total Spending by Category"
)

st.plotly_chart(fig, use_container_width=True)

monthly = (
    filtered.groupby("year_month")["abs_amount"]
    .sum()
    .reset_index()
)

fig2 = px.line(
    monthly,
    x="year_month",
    y="abs_amount",
    title="Monthly Spending Trend"
)

st.plotly_chart(fig2, use_container_width=True)

type_summary = (
    filtered.groupby("spending_type")["abs_amount"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    type_summary,
    names="spending_type",
    values="abs_amount",
    title="Necessary vs Discretionary Spending"
)

st.plotly_chart(fig3, use_container_width=True)

# Forecasting
df["date"] = pd.to_datetime(df["date"])
df["year_month"] = pd.to_datetime(df["year_month"])

expenses = df[df["transaction_type"] == "Expense"]

monthly = (
    expenses.groupby("year_month")["abs_amount"]
    .sum()
    .reset_index()
)

monthly.columns = ["year_month", "monthly_expense"]

monthly["lag_1"] = monthly["monthly_expense"].shift(1)
monthly["lag_2"] = monthly["monthly_expense"].shift(2)
monthly["lag_3"] = monthly["monthly_expense"].shift(3)

monthly["rolling_mean_3"] = monthly["monthly_expense"].rolling(3).mean()
monthly["rolling_std_3"] = monthly["monthly_expense"].rolling(3).std()

monthly = monthly.dropna()

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3"
]

X = monthly[features]
y = monthly["monthly_expense"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

monthly["prediction"] = model.predict(X)

st.subheader("📉 Monthly Expense Forecast")

fig = px.line(
    monthly,
    x="year_month",
    y=["monthly_expense", "prediction"],
    labels={"value": "Amount", "variable": "Type"},
)

st.plotly_chart(fig, use_container_width=True)

# Anomaly
anomalies = expenses.copy()

model = IsolationForest(contamination=0.02, random_state=42)
anomalies["anomaly"] = model.fit_predict(anomalies[["abs_amount"]])

st.subheader("🚨 Unusual Transactions")

anomaly_df = anomalies[anomalies["anomaly"] == -1]

st.dataframe(
    anomaly_df[
        ["date", "category", "amount", "description"]
    ].sort_values("amount", ascending=True)
)

st.subheader("📌 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Expenses", f"₹{expenses['abs_amount'].sum():,.0f}")

col2.metric("Avg Monthly Spend", f"₹{monthly['monthly_expense'].mean():,.0f}")

col3.metric("Anomalies Detected", int(len(anomaly_df)))