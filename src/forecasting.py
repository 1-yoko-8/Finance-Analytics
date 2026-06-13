import pandas as pd

df = pd.read_csv("../data/processed/features.csv")

expenses = df[df["transaction_type"] == "Expense"].copy()

monthly = (
    expenses
    .groupby("year_month")["abs_amount"]
    .sum()
    .reset_index()
)
monthly.columns = ["year_month", "monthly_expense"]  # rename
monthly["year_month"] = pd.to_datetime(monthly["year_month"])  # standard format
monthly = monthly[monthly["year_month"] < "2026-06-01"]  # missed during preprocessing
monthly = monthly[monthly["year_month"] >= "2019-01-01"]

monthly["lag_1"] = monthly["monthly_expense"].shift(1)
monthly["lag_2"] = monthly["monthly_expense"].shift(2)
monthly["lag_3"] = monthly["monthly_expense"].shift(3)

monthly["rolling_mean_3"] = (
    monthly["monthly_expense"]
    .rolling(window=3)
    .mean()
)

monthly["rolling_std_3"] = (
    monthly["monthly_expense"]
    .rolling(window=3)
    .std()
)

monthly = monthly.dropna().reset_index(drop=True)  # first few rows will have missing lag values

# print(monthly.head())    # Test if dataset ready
# print(monthly.shape)



# forecasting starts here
X = monthly[
    [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_std_3"
    ]
]
y = monthly["monthly_expense"]

split_idx = int(len(monthly) * 0.8)

X_train = X.iloc[:split_idx]   # iloc - position; loc - value
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# print("Train:", X_train.shape)
# print("Test :", X_test.shape)
#
# print("\nTrain period:")    # Test
# print(monthly.iloc[:split_idx]["year_month"].min())
# print(monthly.iloc[:split_idx]["year_month"].max())
#
# print("\nTest period:")
# print(monthly.iloc[split_idx:]["year_month"].min())
# print(monthly.iloc[split_idx:]["year_month"].max())



# model training
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Performance
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
) * 100

print(f"MAPE: {mape:.2f}%")

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:,.2f}")
print(f"R²  : {r2:.3f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(
    monthly.iloc[split_idx:]["year_month"],
    y_test,
    label="Actual"
)

plt.plot(
    monthly.iloc[split_idx:]["year_month"],
    y_pred,
    label="Predicted"
)

plt.legend()
plt.title("Monthly Expense Forecast")
plt.show()