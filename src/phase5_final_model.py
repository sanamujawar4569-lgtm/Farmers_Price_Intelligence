# ==============================================================
# PHASE 5 - FINAL MODEL VALIDATION AND SELECTION
# Farmers Price Intelligence
# ==============================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ==============================================================
# PATHS
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGINAL_DATA = os.path.join(
    DATA_DIR,
    "final_dataset.csv"
)

CORRECTED_DATA = os.path.join(
    DATA_DIR,
    "final_dataset_corrected_lag_rolling.csv"
)


# ==============================================================
# HEADER
# ==============================================================

print("=" * 75)
print("PHASE 5 - FINAL MODEL VALIDATION")
print("CORRECTED LAG / ROLLING FEATURES")
print("=" * 75)


# ==============================================================
# 1. LOAD CORRECTED DATASET
# ==============================================================

print("\nLoading corrected dataset:")
print(CORRECTED_DATA)

if not os.path.exists(CORRECTED_DATA):
    raise FileNotFoundError(
        "Corrected dataset not found. Run Phase 4 first."
    )

df = pd.read_csv(CORRECTED_DATA)

print("\nDataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ==============================================================
# 2. DATE PROCESSING
# ==============================================================

print("\n" + "=" * 75)
print("DATE PROCESSING")
print("=" * 75)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

if df["Date"].isna().any():
    raise ValueError("Invalid dates found.")

print("Date range:")
print(df["Date"].min(), "to", df["Date"].max())


# ==============================================================
# 3. REMOVE DUPLICATE COMMODITY + DISTRICT + DATE RECORDS
# ==============================================================

print("\n" + "=" * 75)
print("REMOVING DUPLICATE COMMODITY + DISTRICT + DATE RECORDS")
print("=" * 75)

duplicate_mask = df.duplicated(
    subset=["Commodity", "District", "Date"],
    keep=False
)

duplicate_count = duplicate_mask.sum()

print("Duplicate records before removal:", duplicate_count)

if duplicate_count > 0:

    duplicate_rows = df.loc[
        duplicate_mask,
        ["Commodity", "District", "Date", "Modal_Price"]
    ].sort_values(
        ["Commodity", "District", "Date"]
    )

    duplicate_rows.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "phase5_removed_duplicates.csv"
        ),
        index=False
    )

    print("\nDuplicate records saved for audit.")

    # Keep the first record.
    df = df.drop_duplicates(
        subset=["Commodity", "District", "Date"],
        keep="first"
    )

print(
    "\nRows after duplicate removal:",
    len(df)
)

remaining_duplicates = df.duplicated(
    subset=["Commodity", "District", "Date"]
).sum()

print(
    "Remaining Commodity + District + Date duplicates:",
    remaining_duplicates
)

if remaining_duplicates != 0:
    raise ValueError(
        "Duplicate removal failed."
    )

print("✓ Duplicate validation passed.")


# ==============================================================
# 4. SORT DATA
# ==============================================================

print("\n" + "=" * 75)
print("SORTING DATA")
print("=" * 75)

df = df.sort_values(
    ["Commodity", "District", "Date"]
).reset_index(drop=True)


# ==============================================================
# 5. VERIFY CORRECT LAG / ROLLING FEATURES
# ==============================================================

print("\n" + "=" * 75)
print("VERIFYING LAG / ROLLING FEATURES")
print("=" * 75)

group_cols = ["Commodity", "District"]

# Recalculate from Modal_Price.
df["Verify_Lag_1"] = (
    df.groupby(group_cols)["Modal_Price"]
      .shift(1)
)

df["Verify_Lag_2"] = (
    df.groupby(group_cols)["Modal_Price"]
      .shift(2)
)

df["Verify_Lag_3"] = (
    df.groupby(group_cols)["Modal_Price"]
      .shift(3)
)

df["Verify_Rolling_7"] = (
    df.groupby(group_cols)["Modal_Price"]
      .transform(
          lambda x: x.shift(1).rolling(
              window=7,
              min_periods=1
          ).mean()
      )
)

df["Verify_Rolling_30"] = (
    df.groupby(group_cols)["Modal_Price"]
      .transform(
          lambda x: x.shift(1).rolling(
              window=30,
              min_periods=1
          ).mean()
      )
)


# ==============================================================
# 6. COMPARE STORED VS VERIFIED FEATURES
# ==============================================================

feature_pairs = [
    ("Lag_1", "Verify_Lag_1"),
    ("Lag_2", "Verify_Lag_2"),
    ("Lag_3", "Verify_Lag_3"),
    ("Rolling_7", "Verify_Rolling_7"),
    ("Rolling_30", "Verify_Rolling_30"),
]

verification_results = []

for old_col, verify_col in feature_pairs:

    if old_col not in df.columns:
        print(f"{old_col}: NOT FOUND")
        continue

    valid = (
        df[old_col].notna()
        & df[verify_col].notna()
    )

    if valid.sum() == 0:
        mae = np.nan
        exact = 0
        percentage = 0
    else:

        difference = (
            df.loc[valid, old_col]
            - df.loc[valid, verify_col]
        ).abs()

        mae = difference.mean()

        exact = (
            difference < 0.01
        ).sum()

        percentage = (
            exact / valid.sum()
        ) * 100

    verification_results.append({
        "Feature": old_col,
        "Valid_Rows": int(valid.sum()),
        "Exact_Matches": int(exact),
        "Match_Percentage": percentage,
        "Mean_Absolute_Difference": mae
    })

verification_df = pd.DataFrame(
    verification_results
)

print("\nFeature verification:")
print(
    verification_df.to_string(index=False)
)

verification_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_lag_rolling_verification.csv"
    ),
    index=False
)


# ==============================================================
# 7. USE VERIFIED FEATURES
# ==============================================================
#
# IMPORTANT:
# The verified features are generated from historical prices only.
# Today's Modal_Price is never used.
# ==============================================================

print("\nReplacing lag/rolling features with verified values...")

for old_col, verify_col in feature_pairs:

    if verify_col in df.columns:
        df[old_col] = df[verify_col]

        df.drop(
            columns=[verify_col],
            inplace=True
        )


# ==============================================================
# 8. CHECK MISSING VALUES
# ==============================================================

print("\n" + "=" * 75)
print("MISSING VALUE CHECK")
print("=" * 75)

lag_cols = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]

for col in lag_cols:

    print(
        f"{col}:",
        df[col].isna().sum()
    )


# ==============================================================
# 9. DROP ROWS WITHOUT HISTORICAL FEATURES
# ==============================================================

before = len(df)

df = df.dropna(
    subset=lag_cols
).reset_index(drop=True)

after = len(df)

print(
    "\nRows removed because historical features were unavailable:",
    before - after
)

print(
    "Rows available for modeling:",
    after
)


# ==============================================================
# 10. REMOVE UNSAFE COLUMNS
# ==============================================================

print("\n" + "=" * 75)
print("FEATURE PREPARATION")
print("=" * 75)

# Price_Change may contain today's target.
if "Price_Change" in df.columns:

    print(
        "Removing Price_Change to prevent target leakage."
    )

    df = df.drop(
        columns=["Price_Change"]
    )

# Unit columns are metadata.
drop_columns = [
    "Arrival_Unit",
    "Price_Unit",
    "Rain_mm"
]

for col in drop_columns:

    if col in df.columns:
        df.drop(
            columns=[col],
            inplace=True
        )


# ==============================================================
# 11. TIME BASED SPLIT
# ==============================================================

print("\n" + "=" * 75)
print("TIME BASED TRAIN / TEST SPLIT")
print("=" * 75)

# 80% time split.
split_date = df["Date"].quantile(0.80)

train_df = df[
    df["Date"] <= split_date
].copy()

test_df = df[
    df["Date"] > split_date
].copy()

print("\nSplit date:", split_date)

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


# ==============================================================
# 12. CATEGORICAL ENCODING
# ==============================================================

print("\n" + "=" * 75)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 75)

categorical_columns = [
    "State",
    "Commodity_Group",
    "Commodity",
    "District",
    "Festival"
]

encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    # Fit ONLY on training data.
    train_values = train_df[col].astype(str)

    le.fit(train_values)

    # Handle unseen test categories safely.
    known_values = set(le.classes_)

    train_df[col] = train_df[col].astype(str)

    test_df[col] = test_df[col].astype(str)

    test_df[col] = test_df[col].apply(
        lambda x: x
        if x in known_values
        else le.classes_[0]
    )

    train_df[col] = le.transform(
        train_df[col]
    )

    test_df[col] = le.transform(
        test_df[col]
    )

    encoders[col] = le

    print(
        f"{col}: {len(le.classes_)} categories"
    )


# ==============================================================
# 13. BOOLEAN COLUMNS
# ==============================================================

for data in [train_df, test_df]:

    if "Weekend" in data.columns:

        data["Weekend"] = (
            data["Weekend"]
            .astype(int)
        )


# ==============================================================
# 14. REMOVE DATE
# ==============================================================

train_df = train_df.drop(
    columns=["Date"]
)

test_df = test_df.drop(
    columns=["Date"]
)


# ==============================================================
# 15. TARGET
# ==============================================================

TARGET = "Modal_Price"

X_train_all = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]

X_test_all = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]


# ==============================================================
# 16. RANDOM FOREST FEATURES
# ==============================================================

rf_features = [
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

rf_features = [
    col
    for col in rf_features
    if col in X_train_all.columns
]

X_train_rf = X_train_all[
    rf_features
]

X_test_rf = X_test_all[
    rf_features
]


# ==============================================================
# 17. XGBOOST FEATURES
# ==============================================================
#
# Improved XGBoost uses historical price features too.
# ==============================================================

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
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]

xgb_features = [
    col
    for col in xgb_features
    if col in X_train_all.columns
]

X_train_xgb = X_train_all[
    xgb_features
]

X_test_xgb = X_test_all[
    xgb_features
]


# ==============================================================
# 18. FINAL FEATURE VALIDATION
# ==============================================================

print("\nRF features:")
print(rf_features)

print(
    "Number of RF features:",
    len(rf_features)
)

print("\nXGBoost features:")
print(xgb_features)

print(
    "Number of XGBoost features:",
    len(xgb_features)
)


# ==============================================================
# 19. TRAIN RANDOM FOREST
# ==============================================================

print("\n" + "=" * 75)
print("TRAINING CORRECTED RANDOM FOREST")
print("=" * 75)

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

rf_model.fit(
    X_train_rf,
    y_train
)

rf_pred = rf_model.predict(
    X_test_rf
)

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

rf_r2 = r2_score(
    y_test,
    rf_pred
)

print("\nCorrected Random Forest:")
print("MAE :", round(rf_mae, 4))
print("RMSE:", round(rf_rmse, 4))
print("R2  :", round(rf_r2, 4))


# ==============================================================
# 20. TRAIN IMPROVED XGBOOST
# ==============================================================

print("\n" + "=" * 75)
print("TRAINING CORRECTED IMPROVED XGBOOST")
print("=" * 75)

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_xgb,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test_xgb
)

xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)

print("\nCorrected Improved XGBoost:")
print("MAE :", round(xgb_mae, 4))
print("RMSE:", round(xgb_rmse, 4))
print("R2  :", round(xgb_r2, 4))


# ==============================================================
# 21. MODEL COMPARISON
# ==============================================================

comparison = pd.DataFrame([
    {
        "Model": "Corrected Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2": rf_r2
    },
    {
        "Model": "Corrected Improved XGBoost",
        "MAE": xgb_mae,
        "RMSE": xgb_rmse,
        "R2": xgb_r2
    }
])

print("\n" + "=" * 75)
print("CORRECTED MODEL COMPARISON")
print("=" * 75)

print(
    comparison.to_string(index=False)
)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_corrected_model_comparison.csv"
    ),
    index=False
)


# ==============================================================
# 22. COMMODITY VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("VALIDATION BY COMMODITY")
print("=" * 75)

test_results = test_df.copy()

# Add original categorical names from the test split.
# Use the encoded values to map back.
commodity_encoder = encoders["Commodity"]

test_results["Commodity_Name"] = (
    test_results["Commodity"]
    .astype(int)
    .apply(
        lambda x:
        commodity_encoder.inverse_transform([x])[0]
    )
)

test_results["Actual"] = y_test.values
test_results["Prediction"] = xgb_pred

commodity_results = []

for commodity, group in test_results.groupby(
    "Commodity_Name"
):

    actual = group["Actual"]
    predicted = group["Prediction"]

    commodity_results.append({
        "Commodity": commodity,
        "Rows": len(group),
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "R2": r2_score(
            actual,
            predicted
        )
    })

commodity_df = pd.DataFrame(
    commodity_results
).sort_values(
    "MAE",
    ascending=False
)

print(
    commodity_df.to_string(index=False)
)

commodity_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_xgb_by_commodity.csv"
    ),
    index=False
)


# ==============================================================
# 23. DISTRICT VALIDATION
# ==============================================================

print("\n" + "=" * 75)
print("VALIDATION BY DISTRICT")
print("=" * 75)

district_encoder = encoders["District"]

test_results["District_Name"] = (
    test_results["District"]
    .astype(int)
    .apply(
        lambda x:
        district_encoder.inverse_transform([x])[0]
    )
)

district_results = []

for district, group in test_results.groupby(
    "District_Name"
):

    actual = group["Actual"]
    predicted = group["Prediction"]

    district_results.append({
        "District": district,
        "Rows": len(group),
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "R2": r2_score(
            actual,
            predicted
        )
    })

district_df = pd.DataFrame(
    district_results
).sort_values(
    "MAE",
    ascending=False
)

print(
    district_df.to_string(index=False)
)

district_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_xgb_by_district.csv"
    ),
    index=False
)


# ==============================================================
# 24. SAVE TEST PREDICTIONS
# ==============================================================

prediction_output = test_results[
    [
        "Commodity_Name",
        "District_Name",
        "Actual",
        "Prediction"
    ]
].copy()

prediction_output["Error"] = (
    prediction_output["Prediction"]
    - prediction_output["Actual"]
)

prediction_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase5_final_test_predictions.csv"
    ),
    index=False
)


# ==============================================================
# 25. SELECT FINAL MODEL
# ==============================================================

print("\n" + "=" * 75)
print("FINAL MODEL DECISION")
print("=" * 75)

# Lower MAE is the primary criterion.
# RMSE is secondary.
# R2 is supporting evidence.

if xgb_mae < rf_mae:

    final_model = xgb_model
    final_model_name = "Improved XGBoost"
    final_features = xgb_features

else:

    final_model = rf_model
    final_model_name = "Random Forest"
    final_features = rf_features


print(
    "\nFINAL MODEL:",
    final_model_name
)

print(
    "Final MAE:",
    round(
        min(rf_mae, xgb_mae),
        4
    )
)

print(
    "Final feature count:",
    len(final_features)
)


# ==============================================================
# 26. SAVE FINAL MODEL
# ==============================================================

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_price_prediction_model.pkl"
)

FINAL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "final_label_encoders.pkl"
)

FINAL_FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "final_feature_list.json"
)

joblib.dump(
    final_model,
    FINAL_MODEL_PATH
)

joblib.dump(
    encoders,
    FINAL_ENCODER_PATH
)

with open(
    FINAL_FEATURE_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "model_name": final_model_name,
            "target": TARGET,
            "features": final_features,
            "feature_count": len(final_features)
        },
        f,
        indent=4
    )


# ==============================================================
# 27. SAVE FINAL DATASET
# ==============================================================

FINAL_DATASET = os.path.join(
    DATA_DIR,
    "final_dataset_phase5.csv"
)

df.to_csv(
    FINAL_DATASET,
    index=False
)


# ==============================================================
# 28. FINAL SUMMARY
# ==============================================================

print("\n" + "=" * 75)
print("PHASE 5 COMPLETED")
print("=" * 75)

print("\nFinal model:")
print(final_model_name)

print("\nFinal model saved:")
print(FINAL_MODEL_PATH)

print("\nFinal encoders saved:")
print(FINAL_ENCODER_PATH)

print("\nFinal feature list saved:")
print(FINAL_FEATURE_PATH)

print("\nFinal dataset saved:")
print(FINAL_DATASET)

print("\nGenerated output files:")

print(
    "1.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_removed_duplicates.csv"
    )
)

print(
    "2.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_lag_rolling_verification.csv"
    )
)

print(
    "3.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_corrected_model_comparison.csv"
    )
)

print(
    "4.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_xgb_by_commodity.csv"
    )
)

print(
    "5.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_xgb_by_district.csv"
    )
)

print(
    "6.",
    os.path.join(
        OUTPUT_DIR,
        "phase5_final_test_predictions.csv"
    )

)

print("\n" + "=" * 75)
print("NEXT STEP")
print("=" * 75)

print(
    "Update app.py to use:"
)

print(
    "final_price_prediction_model.pkl"
)

print(
    "final_label_encoders.pkl"
)

print(
    "final_feature_list.json"
)

print("=" * 75)