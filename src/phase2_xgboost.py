import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "final_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 - IMPROVED XGBOOST VALIDATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded.")
print("Rows    :", len(df))
print("Columns :", len(df))


# ============================================================
# DATE PROCESSING
# ============================================================

print("\nProcessing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.sort_values(
    ["Date", "Commodity", "District"]
).reset_index(drop=True)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ============================================================
# BASIC CLEANING
# ============================================================

df = df.drop_duplicates()

# These columns are not model features
columns_to_drop = []

for col in [
    "Arrival_Unit",
    "Price_Unit",
    "Rain_mm"
]:

    if col in df.columns:
        columns_to_drop.append(col)

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

if "Price_Change" in df.columns:

    print(
        "\nPrice_Change found."
    )

    print(
        "Price_Change will NOT be used."
    )

    print(
        "Reason: Price_Change may contain today's target price."
    )


# ============================================================
# IMPORTANT:
# HISTORICAL PRICE FEATURES
# ============================================================

historical_features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]


print("\nChecking historical price features...")

for col in historical_features:

    if col in df.columns:

        print(
            f"{col:12} -> available"
        )

    else:

        print(
            f"{col:12} -> MISSING"
        )


# ============================================================
# TIME BASED SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TIME BASED TRAIN / TEST SPLIT")
print("=" * 70)

split_date = pd.Timestamp("2025-05-27")

train_df = df[
    df["Date"] < split_date
].copy()

test_df = df[
    df["Date"] >= split_date
].copy()

print("\nTraining period:")
print(
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)

print("\nTesting period:")
print(
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)

print("\nTraining rows:", len(train_df))
print("Testing rows :", len(test_df))


# ============================================================
# REMOVE PRICE_CHANGE
# ============================================================

df_features_to_remove = [
    "Modal_Price",
    "Price_Change",
    "Date"
]


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

categorical_columns = [
    "State",
    "Commodity_Group",
    "Commodity",
    "District",
    "Festival"
]

encoders = {}


print("\n" + "=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)


for col in categorical_columns:

    le = LabelEncoder()

    # Fit using complete category vocabulary.
    # This does not use the target variable.
    df[col] = df[col].fillna("Unknown").astype(str)

    le.fit(df[col])

    df[col] = le.transform(
        df[col]
    )

    encoders[col] = le

    print(
        f"{col}: {len(le.classes_)} categories"
    )


# ============================================================
# WEEKEND
# ============================================================

if "Weekend" in df.columns:

    df["Weekend"] = (
        df["Weekend"]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# SPLIT AGAIN AFTER ENCODING
# ============================================================

train_df = df[
    df["Date"] < split_date
].copy()

test_df = df[
    df["Date"] >= split_date
].copy()


# ============================================================
# FEATURE LIST
# ============================================================

xgb_features = [

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

    # Historical price information
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

missing_features = [
    col
    for col in xgb_features
    if col not in df.columns
]

if missing_features:

    print(
        "\nERROR: Missing features:"
    )

    for col in missing_features:
        print(" -", col)

    raise ValueError(
        "Required XGBoost features are missing."
    )


print("\nXGBoost features:")

for i, feature in enumerate(
    xgb_features,
    start=1
):

    print(
        f"{i:2}. {feature}"
    )


# ============================================================
# BUILD X / Y
# ============================================================

X_train = train_df[
    xgb_features
].copy()

y_train = train_df[
    "Modal_Price"
].copy()

X_test = test_df[
    xgb_features
].copy()

y_test = test_df[
    "Modal_Price"
].copy()


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\nChecking missing values...")

print(
    "Training missing:",
    X_train.isna().sum().sum()
)

print(
    "Testing missing:",
    X_test.isna().sum().sum()
)


# XGBoost can handle NaN,
# but fill numeric missing values for stability.

for col in xgb_features:

    if (
        X_train[col].dtype
        != "object"
    ):

        median_value = X_train[
            col
        ].median()

        X_train[col] = X_train[
            col
        ].fillna(
            median_value
        )

        X_test[col] = X_test[
            col
        ].fillna(
            median_value
        )


# ============================================================
# BASELINE XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("BASELINE XGBOOST")
print("=" * 70)

baseline_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=42,

    n_jobs=-1
)


print("\nTraining baseline XGBoost...")

baseline_model.fit(
    X_train,
    y_train
)

print(
    "Baseline XGBoost training completed."
)


baseline_prediction = (
    baseline_model.predict(
        X_test
    )
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_prediction
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_prediction
)


print("\nBaseline XGBoost results")

print(
    "MAE :",
    round(baseline_mae, 4)
)

print(
    "RMSE:",
    round(baseline_rmse, 4)
)

print(
    "R2  :",
    round(baseline_r2, 4)
)


# ============================================================
# IMPROVED XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED XGBOOST")
print("=" * 70)


improved_model = XGBRegressor(

    n_estimators=700,

    learning_rate=0.03,

    max_depth=6,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    reg_alpha=0.05,

    reg_lambda=1.5,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=42,

    n_jobs=-1
)


print(
    "\nTraining improved XGBoost..."
)

improved_model.fit(
    X_train,
    y_train
)

print(
    "Improved XGBoost training completed."
)


# ============================================================
# IMPROVED PREDICTION
# ============================================================

improved_prediction = (
    improved_model.predict(
        X_test
    )
)


improved_mae = mean_absolute_error(
    y_test,
    improved_prediction
)

improved_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        improved_prediction
    )
)

improved_r2 = r2_score(
    y_test,
    improved_prediction
)


print("\nImproved XGBoost results")

print(
    "MAE :",
    round(improved_mae, 4)
)

print(
    "RMSE:",
    round(improved_rmse, 4)
)

print(
    "R2  :",
    round(improved_r2, 4)
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":
        xgb_features,

    "Importance":
        improved_model.feature_importances_

})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("IMPROVED XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "phase2_xgb_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_output = test_df[
    [
        "Date",
        "Commodity",
        "District",
        "Modal_Price"
    ]
].copy()


prediction_output[
    "XGB_Prediction"
] = improved_prediction


prediction_output[
    "Prediction_Error"
] = (
    prediction_output[
        "Modal_Price"
    ]
    -
    prediction_output[
        "XGB_Prediction"
    ]
)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "phase2_xgb_test_predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "XGBoost Baseline",
        "XGBoost Improved"
    ],

    "MAE": [
        baseline_mae,
        improved_mae
    ],

    "RMSE": [
        baseline_rmse,
        improved_rmse
    ],

    "R2": [
        baseline_r2,
        improved_r2
    ]

})


# ============================================================
# LOAD PHASE 1 RF RESULT
# ============================================================

rf_comparison_path = os.path.join(
    OUTPUT_DIR,
    "time_based_model_comparison.csv"
)

if os.path.exists(
    rf_comparison_path
):

    rf_results = pd.read_csv(
        rf_comparison_path
    )

    comparison = pd.concat(
        [
            rf_results,
            comparison
        ],
        ignore_index=True
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "phase2_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("MODEL DECISION")
print("=" * 70)


if improved_r2 > baseline_r2:

    selected_model = improved_model

    print(
        "\nImproved XGBoost performed better than baseline."
    )

else:

    selected_model = baseline_model

    print(
        "\nBaseline XGBoost performed better."
    )


# ============================================================
# SAVE IMPROVED XGBOOST
# ============================================================

final_xgb_path = os.path.join(
    MODEL_DIR,
    "xgboost_improved.pkl"
)

joblib.dump(
    selected_model,
    final_xgb_path
)


print(
    "\nFinal XGBoost saved:"
)

print(
    final_xgb_path
)


# ============================================================
# SAVE ENCODERS
# ============================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoders.pkl"
)

joblib.dump(
    encoders,
    encoder_path
)


print(
    "\nEncoders saved:"
)

print(
    encoder_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 XGBOOST VALIDATION COMPLETED")
print("=" * 70)

print(
    "\nBaseline XGBoost"
)

print(
    "MAE:",
    round(baseline_mae, 4)
)

print(
    "RMSE:",
    round(baseline_rmse, 4)
)

print(
    "R2:",
    round(baseline_r2, 4)
)


print(
    "\nImproved XGBoost"
)

print(
    "MAE:",
    round(improved_mae, 4)
)

print(
    "RMSE:",
    round(improved_rmse, 4)
)

print(
    "R2:",
    round(improved_r2, 4)
)


print("\nGenerated files:")

print(
    "1.",
    final_xgb_path
)

print(
    "2.",
    importance_path
)

print(
    "3.",
    prediction_path
)

print(
    "4.",
    comparison_path
)

print(
    "5.",
    encoder_path
)

print("\n" + "=" * 70)