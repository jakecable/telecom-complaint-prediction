import pandas as pd

# Clean and Prepare the Complaints Data
try:
    # Read the data, specifying the ZIP code column as a string
    df_complaints_raw = pd.read_csv(
        "../data/raw/CGB_-_Consumer_Complaints_Data_20250913.csv", dtype={"Zip": str}
    )
    df_complaints_raw["zip_code"] = df_complaints_raw["Zip"].str.strip().str.slice(0, 5)
    df_complaints_raw.dropna(subset=["zip_code"], inplace=True)
    df_complaints = df_complaints_raw["zip_code"].value_counts().reset_index()
    df_complaints.columns = ["zip_code", "complaint_volume"]
    print("Complaints data processed successfully.")
except FileNotFoundError:
    print("ERROR: CGB_-_Consumer_Complaints_Data_20250913.csv not found.")
    exit()

# Clean and Prepare the Demographics Data
try:
    df_demographics_raw = pd.read_csv(
        " ../data/raw/ACSDP5Y2023.DP05-Data.csv",
        header=0,
        skiprows=[1],
        dtype={"GEO_ID": str},
    )
    df_demographics_raw["zip_code"] = df_demographics_raw["GEO_ID"].str.slice(-5)
    df_demographics = df_demographics_raw[
        ["zip_code", "DP05_0001E", "DP05_0018E", "DP05_0033E"]
    ].copy()
    df_demographics.rename(
        columns={
            "DP05_0001E": "total_population",
            "DP05_0018E": "median_age",
            "DP05_0033E": "total_housing_units",
        },
        inplace=True,
    )
    for col in ["total_population", "median_age", "total_housing_units"]:
        df_demographics[col] = pd.to_numeric(df_demographics[col], errors="coerce")
except FileNotFoundError:
    print("ERROR: ACSDP5Y2023.DP05-Data.csv not found.")
    exit()

try:
    df_broadband_raw = pd.read_csv(
        "../data/raw/bdc_48_fixed_broadband_summary_by_geography_place_D24_03sep2025.csv",
        dtype={"geography_id": str},
    )
    df_broadband_filtered = df_broadband_raw[
        (df_broadband_raw["technology"] == "Any Technology")
        & (df_broadband_raw["biz_res"] == "R")
    ].copy()
    # Keep 'place' as a string with leading zeros
    df_broadband_filtered["place"] = df_broadband_filtered["geography_id"].str.slice(2)
    df_broadband = df_broadband_filtered[["place", "speed_25_3", "speed_100_20"]].copy()
    df_broadband.rename(
        columns={
            "speed_25_3": "pct_broadband_25_3",
            "speed_100_20": "pct_broadband_100_20",
        },
        inplace=True,
    )
except FileNotFoundError:
    print(
        "ERROR: bdc_48_fixed_broadband_summary_by_geography_place_D24_03sep2025.csv not found."
    )
    exit()

# Use the Crosswalk to Map Broadband Data to ZIP Codes
print("\nStep 4: Mapping Broadband Data to ZIP Codes using Geocorr...")
try:
    # *** FIX 1: Add 'stab' (state abbreviation) to the list of columns to read ***
    df_geocorr = pd.read_csv(
        "../data/raw/geocorr.csv",
        skiprows=[1],
        encoding="latin-1",
        dtype={"place": str, "zcta": str, "stab": str},
    )

    # *** FIX 2: Keep 'stab' in the DataFrame ***
    df_geocorr = df_geocorr[["place", "zcta", "stab", "afact"]].copy()
    df_geocorr.rename(columns={"zcta": "zip_code"}, inplace=True)

    # Merge broadband data with the crosswalk using string keys
    df_broadband_by_zip = pd.merge(df_geocorr, df_broadband, on="place", how="left")
    df_broadband_by_zip["weighted_25_3"] = (
        df_broadband_by_zip["pct_broadband_25_3"] * df_broadband_by_zip["afact"]
    )
    df_broadband_by_zip["weighted_100_20"] = (
        df_broadband_by_zip["pct_broadband_100_20"] * df_broadband_by_zip["afact"]
    )

    # Group by both zip_code and state
    df_broadband_final = (
        df_broadband_by_zip.groupby(["zip_code", "stab"])[
            ["weighted_25_3", "weighted_100_20"]
        ]
        .sum()
        .reset_index()
    )
    df_broadband_final.rename(
        columns={
            "weighted_25_3": "avg_pct_broadband_25_3",
            "weighted_100_20": "avg_pct_broadband_100_20",
        },
        inplace=True,
    )
except FileNotFoundError:
    print("ERROR: geocorr.csv not found.")
    exit()

# Merge all cleaned DataFrames
df_unified = df_demographics.copy()
df_unified = pd.merge(df_unified, df_complaints, on="zip_code", how="left")
df_unified = pd.merge(df_unified, df_broadband_final, on="zip_code", how="left")

# Fill NaNs that result from left merges
df_unified["complaint_volume"].fillna(0, inplace=True)
df_unified["avg_pct_broadband_25_3"].fillna(0, inplace=True)
df_unified["avg_pct_broadband_100_20"].fillna(0, inplace=True)

final_columns = [
    "zip_code",
    "stab",
    "complaint_volume",
    "total_population",
    "median_age",
    "total_housing_units",
    "avg_pct_broadband_25_3",
    "avg_pct_broadband_100_20",
]
df_unified = df_unified[final_columns].dropna(
    subset=["stab"]
)  # Drop rows where state is missing

df_unified.to_csv("../data/processed/unified_model_dataset_corrected.csv", index=False)

print("\nFinal Dataset")
print(df_unified.head())
