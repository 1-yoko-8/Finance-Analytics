from src.preprocessing import *
from src.analysis import *

df = load_data("data/transactions.xlsx")
df = preprocess(df)

print(category_spending(df).head())
print()

print(monthly_expense(df).tail())
print()

print(recurring_transactions(df))