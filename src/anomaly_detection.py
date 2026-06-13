import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("../data/processed/features.csv")

# Keep only expenses
expenses = df[df["transaction_type"] == "Expense"].copy()

# Feature for anomaly detection
features = expenses[["abs_amount"]]

# Train model
model = IsolationForest(
    contamination=0.02,
    random_state=42
)

expenses["anomaly"] = model.fit_predict(features)

# Count anomalies
print("Number of anomalies:", (expenses["anomaly"] == -1).sum())

# Extract anomalies
anomalies = expenses[expenses["anomaly"] == -1]

# Show largest anomalies
top_anomalies = (
    anomalies
    .sort_values("abs_amount", ascending=False)
    [["date", "category", "amount", "abs_amount", "description"]]
    .head(20)
)

print("\nTop anomalies:")
print(top_anomalies)

expenses.to_csv(
    "../data/processed/transactions_with_anomalies.csv",
    index=False
)