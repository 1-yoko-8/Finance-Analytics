import pandas as pd

def load_and_preprocess_data(
input_path="../data/processed/cleaned_transactions.csv",
output_path="../data/processed/transactions_preprocessed.csv"
):
    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Date parsing
    df["date"] = pd.to_datetime(df["date"])

    # Remove accidental whitespace in categories
    df["category"] = df["category"].str.strip()

    # Income vs Expense
    df["transaction_type"] = df["amount"].apply(
        lambda x: "Income" if x > 0 else "Expense"
    )

    # Positive transaction amount
    df["abs_amount"] = df["amount"].abs()

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.day_name()

    # Monthly aggregation key
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # Save processed dataset
    df.to_csv(output_path, index=False)

    print("Processed shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data()