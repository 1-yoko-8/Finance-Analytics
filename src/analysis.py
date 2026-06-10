def category_spending(df):
    expenses = df[df["amount"] < 0]

    return (
        expenses.groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
    )

def monthly_expense(df):
    expenses = df[df["amount"] < 0]

    return (
        expenses.groupby("Month")["amount"]
        .sum()
        .abs()
    )

def monthly_income(df):
    income = df[df["amount"] > 0]
    return income.groupby("Month")["amount"].sum()

def monthly_savings(df):
    income = monthly_income(df)
    expense = monthly_expense(df)
    return income - expense

def savings_rate(df):
    income = monthly_income(df)
    savings = monthly_savings(df)
    return (savings / income) * 100

def recurring_transactions(df):
    grouped = df.groupby("description")

    recurring = []

    for desc, group in grouped:

        if len(group) < 3:
            continue

        group = group.sort_values("date")

        diffs = (
            group["date"]
            .diff()
            .dt.days
            .dropna()
        )

        if len(diffs) == 0:
            continue

        avg_gap = diffs.mean()

        if 25 <= avg_gap <= 35:
            recurring.append(desc)

    return recurring