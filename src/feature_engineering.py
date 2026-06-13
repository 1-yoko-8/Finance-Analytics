import pandas as pd

df = pd.read_csv("../data/processed/transactions_preprocessed.csv")

super_category_map = {

    # Food
    "Grocery": "Food",
    "Milk": "Food",
    "Fruits": "Food",
    "Eating out": "Food",
    "Snacks": "Food",
    "Non Veg": "Food",
    "Market": "Food",

    # Housing
    "House Rent": "Housing",
    "House construction": "Housing",
    "House related exp": "Housing",

    # Utilities
    "Bills": "Utilities",
    "Gas": "Utilities",
    "RO water": "Utilities",
    "Mobile recharge": "Utilities",

    # Transport
    "Car fuel": "Transport",
    "OfficeTrans":"Transport",

    # Lifestyle
    "Clothes": "Lifestyle",
    "Entertainment": "Lifestyle",
    "Celebration": "Lifestyle",
    "Gifts": "Lifestyle",
    "New purchase misc": "Lifestyle",
    "Parents wellfare":"Lifestyle",
    "Mobile purchase": "Lifestyle",
    "Tax": "Lifestyle",
    "Misc": "Lifestyle",

    # Health
    "Health": "Health",

    # Education
    "School fees": "Education",
    "Yofiya IITM": "Education",

    # Investments
    "MF/Stock": "Investment",
    "RD": "Investment",
    "Savings": "Investment",
    "LIC premium": "Investment",
    "Gold purchase": "Investment",

    # Loans
    "Loan": "Debt",
    "Loan repayment": "Debt",
    "Jewel loan": "Debt",
    "Seimurai": "Debt",

    # Income
    "Salary": "Income",
    "Otherincome": "Income",
    "Arrears": "Income",
    "DA Arrears": "Income",
    "TA": "Income",
    "SLS":"Income",
    "Melakottaiyur -+":"Income",
    "Feeding": "Income",
    "Deposits": "Income",
    "IOB Madurai": "Income",

}

df["super_category"] = (
    df["category"]
    .map(super_category_map)
    .fillna("Other")
)

def spending_type(category):

    if category in [
        "Food",
        "Housing",
        "Utilities",
        "Transport",
        "Health",
        "Education",
        "Debt"
    ]:
        return "Necessary"

    elif category in [
        "Lifestyle"
    ]:
        return "Discretionary"

    else:
        return "Other"

df["spending_type"] = df["super_category"].apply(spending_type)

categories_to_remove = [
    "Canara bank",
    "IOB bank"
]
df = df[~df["category"].isin(categories_to_remove)]

print(df["super_category"].nunique())
print(df["super_category"].value_counts())

df.to_csv(
    "../data/processed/features.csv",
    index=False
)

print(df.shape)
print(df.columns.tolist())