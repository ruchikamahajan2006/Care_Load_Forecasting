
import pandas as pd
import numpy as np

DATA_PATH = "data/HHS_Unaccompanied_Alien_Children_Program.csv"

def load_and_clean_data(file_path=DATA_PATH):

    # Read CSV
    df = pd.read_csv(file_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Convert Date into datetime
    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    # Remove rows without valid dates
    df = df.dropna(subset=["Date"])

    # Convert all numerical columns
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(
                    ",", "", regex=False
                ),
                errors="coerce"
            )

    # Sort dates from oldest to newest
    df = df.sort_values("Date")

    # Remove duplicate dates
    df = df.drop_duplicates(subset="Date")

    # Set date as index
    df = df.set_index("Date")

    # Create continuous daily dates
    df = df.asfreq("D")

    # Interpolate missing dates using time
    df = df.interpolate(method="time")

    # Fill any remaining boundary missing values
    df = df.ffill().bfill()

    return df

if __name__ == "__main__":

    data = load_and_clean_data()

    print("\nDATASET SUCCESSFULLY CLEANED")
    print("--------------------------------")

    print(data.head())

    print("\nDataset Shape:", data.shape)

    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\nDate Range:")
    print(data.index.min(), "to", data.index.max())