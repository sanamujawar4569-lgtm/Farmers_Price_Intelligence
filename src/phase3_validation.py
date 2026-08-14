import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_improved.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHASE 3 - LAG/ROLLING VALIDATION")
print("IMPROVED XGBOOST - COMMODITY & DISTRICT VALIDATION")
print("=" * 70)


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
# PROCESS DATE
# ============================================================

print("\nProcessing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values(
    [
        "Commodity",
        "District",
        "Date"
    ]
).reset_index(
    drop=True
)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATA VALIDATION")
print("=" * 70)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Missing dates:",
    df["Date"].isna().sum()
)

print(
    "\nCommodity counts:"
)

print(
    df["Commodity"].value_counts()
)

print(
    "\nDistrict counts:"
)

print(
    df["District"].value_counts()
)


# ============================================================
# LAG / ROLLING FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LAG / ROLLING FEATURE VALIDATION")
print("=" * 70)


historical_features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]


for feature in historical_features:

    if feature in df.columns:

        print(
            f"{feature:<12} -> available"
        )

    else:

        print(
            f"{feature:<12} -> MISSING"
        )


# ============================================================
# CHECK CURRENT TARGET CORRELATION
# ============================================================

print("\nChecking correlation with today's Modal_Price...")

correlation_results = []

for feature in historical_features:

    if feature in df.columns:

        corr = df[
            [
                feature,
                "Modal_Price"
            ]
        ].corr().iloc[0, 1]

        correlation_results.append(
            {
                "Feature": feature,
                "Correlation_With_Current_Price": corr
            }
        )

        print(
            f"{feature:<12}: {corr:.6f}"
        )


correlation_df = pd.DataFrame(
    correlation_results
)

correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_lag_rolling_correlations.csv"
    ),
    index=False
)


# ============================================================
# MANUAL LAG CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("MANUAL LAG CONSISTENCY CHECK")
print("=" * 70)

print(
    "\nChecking whether Lag_1 corresponds to previous price "
    "within Commodity + District..."
)


# Create expected previous price

df["Expected_Lag_1"] = (
    df
    .groupby(
        [
            "Commodity",
            "District"
        ]
    )["Modal_Price"]
    .shift(1)
)


lag_check = df[
    [
        "Commodity",
        "District",
        "Date",
        "Modal_Price",
        "Lag_1",
        "Expected_Lag_1"
    ]
].copy()


lag_check["Lag_1_Difference"] = (
    lag_check["Lag_1"]
    -
    lag_check["Expected_Lag_1"]
)


valid_lag_rows = lag_check[
    lag_check["Expected_Lag_1"].notna()
    &
    lag_check["Lag_1"].notna()
]


if len(valid_lag_rows) > 0:

    mean_difference = (
        valid_lag_rows[
            "Lag_1_Difference"
        ]
        .abs()
        .mean()
    )

    print(
        "Average absolute difference:",
        round(
            mean_difference,
            6
        )
    )

    if mean_difference < 0.01:

        print(
            "✓ Lag_1 is consistent with previous price."
        )

    else:

        print(
            "⚠ Lag_1 does not exactly match previous price."
        )

else:

    print(
        "⚠ Not enough valid rows for Lag_1 verification."
    )


lag_check.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_lag1_validation.csv"
    ),
    index=False
)


# ============================================================
# ROLLING FEATURE CHECK
# ============================================================

print("\n" + "=" * 70)
print("ROLLING FEATURE VALIDATION")
print("=" * 70)

print(
    "\nChecking Rolling_7 and Rolling_30..."
)


# Expected historical rolling values.
#
# IMPORTANT:
# shift(1) means today's price is excluded.

df["Expected_Rolling_7"] = (
    df
    .groupby(
        [
            "Commodity",
            "District"
        ]
    )["Modal_Price"]
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


df["Expected_Rolling_30"] = (
    df
    .groupby(
        [
            "Commodity",
            "District"
        ]
    )["Modal_Price"]
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


rolling_results = []


for actual, expected in [

    (
        "Rolling_7",
        "Expected_Rolling_7"
    ),

    (
        "Rolling_30",
        "Expected_Rolling_30"
    )

]:

    if actual in df.columns:

        comparison = df[
            [
                actual,
                expected
            ]
        ].dropna()

        if len(comparison) > 0:

            difference = (
                comparison[actual]
                -
                comparison[expected]
            ).abs()

            mean_difference = (
                difference.mean()
            )

            print(
                f"{actual:<12} "
                f"average difference: "
                f"{mean_difference:.6f}"
            )

            rolling_results.append(
                {
                    "Feature": actual,
                    "Mean_Absolute_Difference":
                        mean_difference
                }
            )


rolling_results_df = pd.DataFrame(
    rolling_results
)

rolling_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_rolling_validation.csv"
    ),
    index=False
)


# ============================================================
# IMPORTANT NOTE
# ============================================================

print(
    "\nNOTE:"
)

print(
    "Rolling features are validated against historical prices."
)

print(
    "The validation uses shift(1), meaning today's Modal_Price "
    "is excluded from the rolling calculation."
)


# ============================================================
# TIME-BASED SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED VALIDATION SET")
print("=" * 70)


split_date = pd.Timestamp(
    "2025-05-27"
)


train_df = df[
    df["Date"] < split_date
].copy()


test_df = df[
    df["Date"] >= split_date
].copy()


print(
    "\nTraining period:"
)

print(
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)


print(
    "\nTesting period:"
)

print(
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)


print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Testing rows :",
    len(test_df)
)


# ============================================================
# LOAD IMPROVED XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("LOADING IMPROVED XGBOOST")
print("=" * 70)


print(
    "\nModel:"
)

print(
    MODEL_PATH
)


xgb_model = joblib.load(
    MODEL_PATH
)


print(
    "\nImproved XGBoost loaded successfully."
)


# ============================================================
# GET MODEL FEATURES
# ============================================================

if hasattr(
    xgb_model,
    "feature_names_in_"
):

    model_features = list(
        xgb_model.feature_names_in_
    )

else:

    model_features = [
        "State",
        "Commodity_Group",
        "Commodity",
        "Arrival",
        "District",
        "Max_Temp",
        "Min_Temp",
        "Rainfall_mm",
        "WindSpeed",
        "Festival",
        "Holiday",
        "Month",
        "Year",
        "Day",
        "Weekend",
        "DayOfWeek",
        "Quarter",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Rolling_7",
        "Rolling_30"
    ]


print(
    "\nModel features:"
)

for i, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# LOAD ENCODERS
# ============================================================

encoder_path = os.path.join(
    BASE_DIR,
    "models",
    "label_encoders.pkl"
)


encoders = joblib.load(
    encoder_path
)


print(
    "\nEncoders loaded."
)


# ============================================================
# ENCODE DATA
# ============================================================

categorical_columns = [
    "State",
    "Commodity_Group",
    "Commodity",
    "District",
    "Festival"
]


encoded_train = train_df.copy()
encoded_test = test_df.copy()


for column in categorical_columns:

    encoder = encoders[column]

    encoded_train[column] = (
        encoder.transform(
            encoded_train[column]
        )
    )

    encoded_test[column] = (
        encoder.transform(
            encoded_test[column]
        )
    )


# ============================================================
# MODEL INPUT
# ============================================================

X_test = encoded_test[
    model_features
].copy()


y_test = encoded_test[
    "Modal_Price"
].copy()


# ============================================================
# OVERALL XGBOOST VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL IMPROVED XGBOOST VALIDATION")
print("=" * 70)


overall_prediction = (
    xgb_model.predict(
        X_test
    )
)


overall_mae = mean_absolute_error(
    y_test,
    overall_prediction
)


overall_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        overall_prediction
    )
)


overall_r2 = r2_score(
    y_test,
    overall_prediction
)


print(
    "\nOverall MAE :",
    round(
        overall_mae,
        4
    )
)


print(
    "Overall RMSE:",
    round(
        overall_rmse,
        4
    )
)


print(
    "Overall R2  :",
    round(
        overall_r2,
        4
    )
)


# ============================================================
# SAVE OVERALL PREDICTIONS
# ============================================================

overall_predictions_df = encoded_test[
    [
        "Date",
        "Commodity",
        "District",
        "Modal_Price"
    ]
].copy()


overall_predictions_df[
    "Predicted_Price"
] = overall_prediction


overall_predictions_df[
    "Absolute_Error"
] = (
    overall_predictions_df[
        "Modal_Price"
    ]
    -
    overall_predictions_df[
        "Predicted_Price"
    ]
).abs()


overall_predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_overall_predictions.csv"
    ),
    index=False
)


# ============================================================
# COMMODITY VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION BY COMMODITY")
print("=" * 70)


commodity_results = []


for commodity in sorted(
    encoded_test["Commodity"].unique()
):

    subset = encoded_test[
        encoded_test["Commodity"]
        == commodity
    ].copy()


    if len(subset) < 10:

        continue


    X = subset[
        model_features
    ]


    y = subset[
        "Modal_Price"
    ]


    prediction = (
        xgb_model.predict(
            X
        )
    )


    mae = mean_absolute_error(
        y,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y,
            prediction
        )
    )


    r2 = r2_score(
        y,
        prediction
    )


    original_name = (
        encoders[
            "Commodity"
        ].inverse_transform(
            [commodity]
        )[0]
    )


    commodity_results.append(
        {
            "Commodity":
                original_name,

            "Rows":
                len(subset),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2
        }
    )


commodity_results_df = pd.DataFrame(
    commodity_results
)


commodity_results_df = (
    commodity_results_df
    .sort_values(
        "MAE"
    )
)


print(
    commodity_results_df.to_string(
        index=False
    )
)


commodity_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_xgb_by_commodity.csv"
    ),
    index=False
)


# ============================================================
# DISTRICT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION BY DISTRICT")
print("=" * 70)


district_results = []


for district in sorted(
    encoded_test["District"].unique()
):

    subset = encoded_test[
        encoded_test["District"]
        == district
    ].copy()


    if len(subset) < 10:

        continue


    X = subset[
        model_features
    ]


    y = subset[
        "Modal_Price"
    ]


    prediction = (
        xgb_model.predict(
            X
        )
    )


    mae = mean_absolute_error(
        y,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y,
            prediction
        )
    )


    r2 = r2_score(
        y,
        prediction
    )


    original_name = (
        encoders[
            "District"
        ].inverse_transform(
            [district]
        )[0]
    )


    district_results.append(
        {
            "District":
                original_name,

            "Rows":
                len(subset),

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2
        }
    )


district_results_df = pd.DataFrame(
    district_results
)


district_results_df = (
    district_results_df
    .sort_values(
        "MAE"
    )
)


print(
    district_results_df.to_string(
        index=False
    )
)


district_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_xgb_by_district.csv"
    ),
    index=False
)


# ============================================================
# WORST COMMODITIES
# ============================================================

print("\n" + "=" * 70)
print("WEAK COMMODITIES")
print("=" * 70)


if not commodity_results_df.empty:

    worst_commodities = (
        commodity_results_df
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(3)
    )

    print(
        worst_commodities.to_string(
            index=False
        )
    )


# ============================================================
# WORST DISTRICTS
# ============================================================

print("\n" + "=" * 70)
print("WEAK DISTRICTS")
print("=" * 70)


if not district_results_df.empty:

    worst_districts = (
        district_results_df
        .sort_values(
            "MAE",
            ascending=False
        )
        .head(5)
    )

    print(
        worst_districts.to_string(
            index=False
        )
    )


# ============================================================
# FINAL PHASE 3 DECISION
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 DECISION")
print("=" * 70)


print(
    "\nOverall Improved XGBoost:"
)

print(
    "MAE :",
    round(
        overall_mae,
        2
    )
)

print(
    "RMSE:",
    round(
        overall_rmse,
        2
    )
)

print(
    "R2  :",
    round(
        overall_r2,
        4
    )
)


if overall_r2 >= 0.90:

    print(
        "\n✓ Overall XGBoost performance is strong."
    )

else:

    print(
        "\n⚠ Overall XGBoost performance needs improvement."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "Model":
                "Improved XGBoost",

            "MAE":
                overall_mae,

            "RMSE":
                overall_rmse,

            "R2":
                overall_r2,

            "Train_Rows":
                len(train_df),

            "Test_Rows":
                len(test_df)
        }
    ]
)


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase3_xgb_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 COMPLETED")
print("=" * 70)


print(
    "\nGenerated files:"
)

print(
    "1. outputs\\phase3_lag_rolling_correlations.csv"
)

print(
    "2. outputs\\phase3_lag1_validation.csv"
)

print(
    "3. outputs\\phase3_rolling_validation.csv"
)

print(
    "4. outputs\\phase3_overall_predictions.csv"
)

print(
    "5. outputs\\phase3_xgb_by_commodity.csv"
)

print(
    "6. outputs\\phase3_xgb_by_district.csv"
)

print(
    "7. outputs\\phase3_xgb_summary.csv"
)

print(
    "\nPhase 3 validation finished."
)

print("=" * 70)