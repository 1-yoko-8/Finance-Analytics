import pandas as pd

def load_data(path):
    return pd.read_excel(path)

def preprocess(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["Month"] = df["date"].dt.to_period("M")
    df["Year"] = df["date"].dt.year
    df["Weekday"] = df["date"].dt.day_name()

    return df