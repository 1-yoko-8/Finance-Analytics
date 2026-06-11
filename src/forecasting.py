import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ----------------------------
# 1. Sample Data (replace with your CSV)
# ----------------------------
# Assume columns: date, amount

data = {
    "date": pd.date_range(start="2020-01-01", periods=84, freq="M"),
    "amount": abs(1000 + 200 * pd.np.sin(range(84)) + pd.np.random.randn(84) * 100)
}

df = pd.DataFrame(data)

# ----------------------------
# 2. Preprocessing
# ----------------------------
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
monthly.index = monthly.index.to_timestamp()

# ----------------------------
# 3. Train ARIMA model
# ----------------------------
model = ARIMA(monthly, order=(2, 1, 2))  # simple random starter config
model_fit = model.fit()

# ----------------------------
# 4. Forecast next 12 months
# ----------------------------
forecast = model_fit.forecast(steps=12)

# ----------------------------
# 5. Plot results
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(monthly, label="Historical Expenses")
plt.plot(forecast.index, forecast, label="Forecast", linestyle="--")

plt.title("Monthly Expense Forecast")
plt.legend()
plt.show()