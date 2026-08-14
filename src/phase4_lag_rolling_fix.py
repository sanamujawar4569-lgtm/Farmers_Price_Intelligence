import os
import pandas as pd
import numpy as np


# ============================================================
# PHASE 4
# DEEP VALIDATION AND CORRECTION OF LAG / ROLLING FEATURES
# ============================================================

print("=" * 70)
print("PHASE 4 - DEEP LAG / ROLLING FEATURE VALIDATION")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "final_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

CORRECTED_DATA_PATH = os.path.join(
    DATA_PATH.replace(
        ".csv",
        "_corrected_lag_rolling.csv"
    )
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)

required_columns = [
    "Date",
    "Commodity",
    "District",
    "Modal_Price",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:

    print("\nERROR: Missing required columns:")
    print(missing_required)

    raise SystemExit


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nDuplicate rows:")

duplicates = df.duplicated().sum()

print(duplicates)


# ============================================================
# CHECK NULL DATES
# ============================================================

null_dates = df["Date"].isna().sum()

print("\nInvalid dates:", null_dates)

if null_dates > 0:

    print(
        "WARNING: Invalid dates detected."
    )


# ============================================================
# SORT DATA
# ============================================================

print("\n" + "=" * 70)
print("SORTING DATA")
print("=" * 70)

group_columns = [
    "Commodity",
    "District"
]

df = df.sort_values(
    group_columns + ["Date"]
).reset_index(drop=True)


# ============================================================
# CHECK DUPLICATE DATE WITHIN GROUP
# ============================================================

print("\nChecking duplicate dates inside")
print("Commodity + District groups...")

duplicate_group_dates = (
    df.groupby(
        group_columns + ["Date"]
    )
    .size()
    .reset_index(name="count")
)

duplicate_group_dates = (
    duplicate_group_dates[
        duplicate_group_dates["count"] > 1
    ]
)

print(
    "Duplicate Commodity + District + Date rows:",
    len(duplicate_group_dates)
)


# ============================================================
# REBUILD HISTORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("RECALCULATING HISTORICAL FEATURES")
print("=" * 70)

grouped_price = df.groupby(
    group_columns
)["Modal_Price"]


# ------------------------------------------------------------
# Lag 1
# ------------------------------------------------------------

df["Correct_Lag_1"] = (
    grouped_price
    .shift(1)
)


# ------------------------------------------------------------
# Lag 2
# ------------------------------------------------------------

df["Correct_Lag_2"] = (
    grouped_price
    .shift(2)
)


# ------------------------------------------------------------
# Lag 3
# ------------------------------------------------------------

df["Correct_Lag_3"] = (
    grouped_price
    .shift(3)
)


# ============================================================
# CORRECT ROLLING FEATURES
# ============================================================

# IMPORTANT:
#
# shift(1) means today's Modal_Price is NOT included.
#
# Therefore there is no target leakage.
# ============================================================

df["Correct_Rolling_7"] = (
    df.groupby(group_columns)["Modal_Price"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )
)


df["Correct_Rolling_30"] = (
    df.groupby(group_columns)["Modal_Price"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=30,
            min_periods=1
        )
        .mean()
    )
)


# ============================================================
# COMPARE EXISTING FEATURES
# ============================================================

print("\n" + "=" * 70)
print("OLD vs CORRECT FEATURE COMPARISON")
print("=" * 70)


features = [
    ("Lag_1", "Correct_Lag_1"),
    ("Lag_2", "Correct_Lag_2"),
    ("Lag_3", "Correct_Lag_3"),
    ("Rolling_7", "Correct_Rolling_7"),
    ("Rolling_30", "Correct_Rolling_30")
]


comparison_results = []


for old_col, correct_col in features:

    valid = (
        df[old_col].notna()
        &
        df[correct_col].notna()
    )

    differences = (
        df.loc[valid, old_col]
        -
        df.loc[valid, correct_col]
    )

    absolute_difference = (
        differences.abs()
    )

    exact_matches = (
        np.isclose(
            df.loc[valid, old_col],
            df.loc[valid, correct_col],
            rtol=1e-5,
            atol=1e-5
        )
        .sum()
    )

    total_valid = valid.sum()

    match_percentage = (
        exact_matches /
        total_valid *
        100
        if total_valid > 0
        else 0
    )

    mean_difference = (
        absolute_difference.mean()
        if len(absolute_difference) > 0
        else np.nan
    )

    max_difference = (
        absolute_difference.max()
        if len(absolute_difference) > 0
        else np.nan
    )

    print("\n", old_col)

    print(
        "Valid rows:",
        total_valid
    )

    print(
        "Exact matches:",
        exact_matches
    )

    print(
        "Match percentage:",
        round(
            match_percentage,
            2
        ),
        "%"
    )

    print(
        "Mean absolute difference:",
        round(
            mean_difference,
            4
        )
    )

    print(
        "Maximum difference:",
        round(
            max_difference,
            4
        )
    )

    comparison_results.append({

        "Feature": old_col,

        "Valid_Rows": total_valid,

        "Exact_Matches": exact_matches,

        "Match_Percentage":
            match_percentage,

        "Mean_Absolute_Difference":
            mean_difference,

        "Maximum_Difference":
            max_difference

    })


comparison_df = pd.DataFrame(
    comparison_results
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "phase4_lag_rolling_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)


leakage_results = []


for feature in [
    "Correct_Lag_1",
    "Correct_Lag_2",
    "Correct_Lag_3",
    "Correct_Rolling_7",
    "Correct_Rolling_30"
]:

    current_price = df[
        "Modal_Price"
    ]

    historical_feature = df[
        feature
    ]

    valid = (
        current_price.notna()
        &
        historical_feature.notna()
    )

    correlation = (
        current_price[valid]
        .corr(
            historical_feature[valid]
        )
    )

    leakage_results.append({

        "Feature": feature,

        "Correlation_With_Today":
            correlation

    })

    print(
        feature,
        "correlation with today's price:",
        round(
            correlation,
            6
        )
    )


leakage_df = pd.DataFrame(
    leakage_results
)


leakage_path = os.path.join(
    OUTPUT_DIR,
    "phase4_historical_feature_leakage_check.csv"
)

leakage_df.to_csv(
    leakage_path,
    index=False
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLE ROWS")
print("=" * 70)


example_columns = [

    "Date",
    "Commodity",
    "District",
    "Modal_Price",

    "Lag_1",
    "Correct_Lag_1",

    "Lag_2",
    "Correct_Lag_2",

    "Lag_3",
    "Correct_Lag_3",

    "Rolling_7",
    "Correct_Rolling_7",

    "Rolling_30",
    "Correct_Rolling_30"

]


print(
    df[
        example_columns
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# REPLACE OLD FEATURES WITH CORRECT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING CORRECTED DATASET")
print("=" * 70)


df["Lag_1"] = df[
    "Correct_Lag_1"
]

df["Lag_2"] = df[
    "Correct_Lag_2"
]

df["Lag_3"] = df[
    "Correct_Lag_3"
]

df["Rolling_7"] = df[
    "Correct_Rolling_7"
]

df["Rolling_30"] = df[
    "Correct_Rolling_30"
]


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "Correct_Lag_1",
        "Correct_Lag_2",
        "Correct_Lag_3",
        "Correct_Rolling_7",
        "Correct_Rolling_30"
    ]
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values in corrected features:")

for feature in [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]:

    print(
        feature,
        ":",
        df[feature].isna().sum()
    )


# ============================================================
# SAVE CORRECTED DATASET
# ============================================================

df.to_csv(
    CORRECTED_DATA_PATH,
    index=False
)


print("\nCorrected dataset saved:")
print(
    CORRECTED_DATA_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)

print(
    "\nComparison file:"
)

print(
    comparison_path
)

print(
    "\nLeakage check:"
)

print(
    leakage_path
)

print(
    "\nCorrected dataset:"
)

print(
    CORRECTED_DATA_PATH
)

print("\n" + "=" * 70)
print("PHASE 4 FEATURE VALIDATION COMPLETED")
print("=" * 70)