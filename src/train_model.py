# ==========================================================
# FARMERS PRICE INTELLIGENCE
# TIME-BASED ML MODEL TRAINING
# Random Forest + XGBoost
# ==========================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT PATHS
# ==========================================================

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


# ==========================================================
# SETTINGS
# ==========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================================
# FEATURE DEFINITIONS
# ==========================================================

CATEGORICAL_COLUMNS = [
    "State",
    "Commodity_Group",
    "Commodity",
    "District",
    "Festival"
]


# ----------------------------------------------------------
# Features that are safe for RF
# ----------------------------------------------------------

RF_FEATURES = [
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


# ----------------------------------------------------------
# XGBoost features
# ----------------------------------------------------------

XGB_FEATURES = [
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
    "Quarter"
]


TARGET = "Modal_Price"


# ==========================================================
# HELPER FUNCTION
# ==========================================================

def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================================
# START
# ==========================================================

print()
print("=" * 70)
print("FARMERS PRICE INTELLIGENCE - TIME BASED MODEL TRAINING")
print("=" * 70)


# ==========================================================
# LOAD DATA
# ==========================================================

print()
print("Loading dataset:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print()
print("Dataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = list(
    set(
        RF_FEATURES
        + XGB_FEATURES
        + CATEGORICAL_COLUMNS
        + [TARGET, "Date"]
    )
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print()
    print("ERROR: Missing columns:")
    print(missing_columns)

    raise ValueError(
        "Required columns are missing from final_dataset.csv"
    )


# ==========================================================
# DATE PROCESSING
# ==========================================================

print()
print("Processing dates...")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", TARGET]
)

df = df.sort_values(
    "Date"
).reset_index(
    drop=True
)

print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ==========================================================
# CHECK DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()

print()
print("Duplicate rows:", duplicates)

if duplicates > 0:

    df = df.drop_duplicates().reset_index(
        drop=True
    )

    print(
        "Duplicates removed."
    )


# ==========================================================
# CHECK TARGET
# ==========================================================

print()
print("Target statistics:")

print(
    df[TARGET].describe()
)


# ==========================================================
# DATA LEAKAGE CHECK
# ==========================================================

print()
print("=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)


# ----------------------------------------------------------
# Check Price_Change
# ----------------------------------------------------------

if "Price_Change" in df.columns:

    print()
    print(
        "WARNING: Price_Change exists in dataset."
    )

    print(
        "It will NOT be used for the final RF model."
    )

    print(
        "Reason: if Price_Change was calculated using"
    )

    print(
        "today's Modal_Price, it would leak the target."
    )


# ----------------------------------------------------------
# Check correlations
# ----------------------------------------------------------

numeric_check_features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]

print()
print("Correlation with Modal_Price:")

for col in numeric_check_features:

    if col in df.columns:

        correlation = df[
            [col, TARGET]
        ].corr().iloc[0, 1]

        print(
            f"{col:12s}: {correlation:.6f}"
        )


# ==========================================================
# REMOVE CURRENT-TARGET LEAKAGE FEATURES
# ==========================================================

# IMPORTANT:
#
# Price_Change is intentionally excluded.
#
# If Rolling_7 / Rolling_30 were created from a window
# including the current Modal_Price, they could also leak.
#
# We cannot safely reconstruct them here without knowing
# exactly how the original feature-engineering notebook
# calculated them.
#
# Therefore we retain them only after checking that their
# calculation in feature engineering is shifted.
#
# For now we keep them because your existing project
# depends on them.


# ==========================================================
# TIME BASED TRAIN / TEST SPLIT
# ==========================================================

print()
print("=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)


unique_dates = np.sort(
    df["Date"].unique()
)

split_index = int(
    len(unique_dates) * (1 - TEST_SIZE)
)

train_end_date = unique_dates[
    split_index - 1
]

test_start_date = unique_dates[
    split_index
]


train_df = df[
    df["Date"] <= train_end_date
].copy()


test_df = df[
    df["Date"] >= test_start_date
].copy()


print()
print("Training period:")
print(
    train_df["Date"].min(),
    "to",
    train_df["Date"].max()
)

print()
print("Testing period:")
print(
    test_df["Date"].min(),
    "to",
    test_df["Date"].max()
)

print()
print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))


# ==========================================================
# VERIFY NO DATE OVERLAP
# ==========================================================

if train_df["Date"].max() >= test_df["Date"].min():

    raise ValueError(
        "ERROR: Training and testing dates overlap!"
    )

print()
print(
    "✓ No date overlap between training and testing."
)


# ==========================================================
# ENCODING
# ==========================================================

print()
print("=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)


encoders = {}

for column in CATEGORICAL_COLUMNS:

    encoder = LabelEncoder()

    # Fit ONLY on training data
    train_df[column] = encoder.fit_transform(
        train_df[column].astype(str)
    )

    # Check unseen categories
    test_values = test_df[column].astype(str)

    unseen_values = set(
        test_values.unique()
    ) - set(
        encoder.classes_
    )

    if unseen_values:

        print()
        print(
            f"WARNING: Unseen {column} values in test:"
        )

        print(
            unseen_values
        )

        # For this project we expect categories
        # to be known across the full time period.
        raise ValueError(
            f"Unseen categories found in {column}: "
            f"{unseen_values}"
        )

    test_df[column] = encoder.transform(
        test_values
    )

    encoders[column] = encoder


# ==========================================================
# CONVERT BOOLEAN / INTEGER FEATURES
# ==========================================================

if "Weekend" in train_df.columns:

    train_df["Weekend"] = train_df[
        "Weekend"
    ].astype(int)

    test_df["Weekend"] = test_df[
        "Weekend"
    ].astype(int)


# ==========================================================
# BUILD FEATURES
# ==========================================================

print()
print("=" * 70)
print("BUILDING MODEL FEATURES")
print("=" * 70)


X_rf_train = train_df[
    RF_FEATURES
].copy()

X_rf_test = test_df[
    RF_FEATURES
].copy()


X_xgb_train = train_df[
    XGB_FEATURES
].copy()

X_xgb_test = test_df[
    XGB_FEATURES
].copy()


y_train = train_df[
    TARGET
].astype(float)

y_test = test_df[
    TARGET
].astype(float)


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

print()
print("Checking missing values...")

for feature_df in [
    X_rf_train,
    X_rf_test,
    X_xgb_train,
    X_xgb_test
]:

    if feature_df.isnull().sum().sum() > 0:

        raise ValueError(
            "Missing values found in model features."
        )


# ==========================================================
# RANDOM FOREST
# ==========================================================

print()
print("=" * 70)
print("RANDOM FOREST")
print("=" * 70)

print()
print(
    "RF features:",
    RF_FEATURES
)

print(
    "Number of RF features:",
    len(RF_FEATURES)
)


rf_model = RandomForestRegressor(

    n_estimators=200,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    max_features="sqrt"

)


print()
print("Training Random Forest...")

rf_model.fit(
    X_rf_train,
    y_train
)

print(
    "Random Forest training completed."
)


# ==========================================================
# RF EVALUATION
# ==========================================================

rf_result = evaluate_model(
    rf_model,
    X_rf_test,
    y_test,
    "Random Forest - Time Based Test"
)


# ==========================================================
# RF FEATURE IMPORTANCE
# ==========================================================

rf_importance = pd.DataFrame({

    "Feature": RF_FEATURES,

    "Importance":
        rf_model.feature_importances_

})


rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)


print()
print(
    "Random Forest Feature Importance:"
)

print(
    rf_importance.to_string(
        index=False
    )
)


rf_importance_path = os.path.join(
    OUTPUT_DIR,
    "rf_time_based_feature_importance.csv"
)

rf_importance.to_csv(
    rf_importance_path,
    index=False
)


# ==========================================================
# SAVE RF MODEL
# ==========================================================

rf_model_path = os.path.join(
    MODEL_DIR,
    "price_prediction_model.pkl"
)

joblib.dump(
    rf_model,
    rf_model_path
)

print()
print(
    "Random Forest saved:"
)

print(
    rf_model_path
)


# ==========================================================
# XGBOOST
# ==========================================================

print()
print("=" * 70)
print("XGBOOST")
print("=" * 70)

print()
print(
    "XGBoost features:",
    XGB_FEATURES
)

print(
    "Number of XGBoost features:",
    len(XGB_FEATURES)
)


xgb_model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=8,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


print()
print("Training XGBoost...")

xgb_model.fit(
    X_xgb_train,
    y_train
)

print(
    "XGBoost training completed."
)


# ==========================================================
# XGB EVALUATION
# ==========================================================

xgb_result = evaluate_model(
    xgb_model,
    X_xgb_test,
    y_test,
    "XGBoost - Time Based Test"
)


# ==========================================================
# XGB FEATURE IMPORTANCE
# ==========================================================

xgb_importance = pd.DataFrame({

    "Feature": XGB_FEATURES,

    "Importance":
        xgb_model.feature_importances_

})


xgb_importance = xgb_importance.sort_values(
    "Importance",
    ascending=False
)


print()
print(
    "XGBoost Feature Importance:"
)

print(
    xgb_importance.to_string(
        index=False
    )
)


xgb_importance_path = os.path.join(
    OUTPUT_DIR,
    "xgb_time_based_feature_importance.csv"
)

xgb_importance.to_csv(
    xgb_importance_path,
    index=False
)


# ==========================================================
# SAVE XGBOOST MODEL
# ==========================================================

xgb_model_path = os.path.join(
    MODEL_DIR,
    "xgboost_weather.pkl"
)

joblib.dump(
    xgb_model,
    xgb_model_path
)

print()
print(
    "XGBoost saved:"
)

print(
    xgb_model_path
)


# ==========================================================
# SAVE ENCODERS
# ==========================================================

encoder_path = os.path.join(
    MODEL_DIR,
    "label_encoders.pkl"
)

joblib.dump(
    encoders,
    encoder_path
)

print()
print(
    "Encoders saved:"
)

print(
    encoder_path
)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

comparison = pd.DataFrame([
    rf_result,
    xgb_result
])


comparison_path = os.path.join(
    OUTPUT_DIR,
    "time_based_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================================
# SAVE TEST PREDICTIONS
# ==========================================================

rf_predictions = rf_model.predict(
    X_rf_test
)

xgb_predictions = xgb_model.predict(
    X_xgb_test
)


prediction_results = test_df[
    [
        "Date",
        "State",
        "Commodity_Group",
        "Commodity",
        "District",
        TARGET
    ]
].copy()


prediction_results[
    "RF_Prediction"
] = rf_predictions


prediction_results[
    "XGB_Prediction"
] = xgb_predictions


prediction_results[
    "RF_Error"
] = (
    prediction_results[TARGET]
    -
    prediction_results["RF_Prediction"]
)


prediction_results[
    "XGB_Error"
] = (
    prediction_results[TARGET]
    -
    prediction_results["XGB_Prediction"]
)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "time_based_test_predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()
print("=" * 70)
print("MODEL TRAINING COMPLETED")
print("=" * 70)

print()

print("Random Forest")
print(
    f"Features : {len(RF_FEATURES)}"
)
print(
    f"MAE      : {rf_result['MAE']:.4f}"
)
print(
    f"RMSE     : {rf_result['RMSE']:.4f}"
)
print(
    f"R2       : {rf_result['R2']:.4f}"
)

print()

print("XGBoost")
print(
    f"Features : {len(XGB_FEATURES)}"
)
print(
    f"MAE      : {xgb_result['MAE']:.4f}"
)
print(
    f"RMSE     : {xgb_result['RMSE']:.4f}"
)
print(
    f"R2       : {xgb_result['R2']:.4f}"
)

print()

print("Generated files:")

print(
    "1.",
    rf_model_path
)

print(
    "2.",
    xgb_model_path
)

print(
    "3.",
    encoder_path
)

print(
    "4.",
    comparison_path
)

print(
    "5.",
    rf_importance_path
)

print(
    "6.",
    xgb_importance_path
)

print(
    "7.",
    prediction_path
)

print()

print("=" * 70)
print("PHASE 1 COMPLETED")
print("=" * 70)