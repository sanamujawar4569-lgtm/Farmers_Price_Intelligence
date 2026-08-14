import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "final_dataset_phase5.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("PHASE 6 - FINAL MODEL IMPROVEMENT & PRODUCTION VALIDATION")
print("=" * 78)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading Phase 5 dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Rows    :", len(df))
print("Columns :", len(df))


# ============================================================
# 2. DATE PROCESSING
# ============================================================

print("\n" + "=" * 78)
print("DATE VALIDATION")
print("=" * 78)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("Date range:")
print(df["Date"].min(), "to", df["Date"].max())

print("Invalid dates:", df["Date"].isna().sum())

df = df.sort_values(
    ["Commodity", "District", "Date"]
).reset_index(drop=True)


# ============================================================
# 3. OBSERVATION LEVEL VALIDATION
# ============================================================

print("\n" + "=" * 78)
print("OBSERVATION LEVEL VALIDATION")
print("=" * 78)

group_cols = ["Commodity", "District", "Date"]

duplicate_count = df.duplicated(
    subset=group_cols
).sum()

print(
    "Commodity + District + Date duplicates:",
    duplicate_count
)

if duplicate_count == 0:
    print("✓ Observation level is unique.")
else:
    print("⚠ Duplicate observation level detected.")


# ============================================================
# 4. HISTORICAL FEATURE VERIFICATION
# ============================================================

print("\n" + "=" * 78)
print("LAG / ROLLING FEATURE VERIFICATION")
print("=" * 78)

# IMPORTANT:
# The current dataset is one observation per
# Commodity + District + Date.
#
# Therefore historical features are calculated within
# Commodity + District groups.

df["Verified_Lag_1"] = (
    df.groupby(["Commodity", "District"])["Modal_Price"]
      .shift(1)
)

df["Verified_Lag_2"] = (
    df.groupby(["Commodity", "District"])["Modal_Price"]
      .shift(2)
)

df["Verified_Lag_3"] = (
    df.groupby(["Commodity", "District"])["Modal_Price"]
      .shift(3)
)

df["Verified_Rolling_7"] = (
    df.groupby(["Commodity", "District"])["Modal_Price"]
      .transform(
          lambda x: x.shift(1).rolling(
              window=7,
              min_periods=7
          ).mean()
      )
)

df["Verified_Rolling_30"] = (
    df.groupby(["Commodity", "District"])["Modal_Price"]
      .transform(
          lambda x: x.shift(1).rolling(
              window=30,
              min_periods=30
          ).mean()
      )
)


historical_features = [
    ("Lag_1", "Verified_Lag_1"),
    ("Lag_2", "Verified_Lag_2"),
    ("Lag_3", "Verified_Lag_3"),
    ("Rolling_7", "Verified_Rolling_7"),
    ("Rolling_30", "Verified_Rolling_30")
]


verification_rows = []


for old_col, verified_col in historical_features:

    valid = (
        df[old_col].notna()
        & df[verified_col].notna()
    )

    if valid.sum() == 0:
        exact_matches = 0
        match_percentage = 0
        mean_difference = np.nan
    else:

        difference = (
            df.loc[valid, old_col]
            - df.loc[valid, verified_col]
        ).abs()

        exact_matches = (
            difference < 1e-6
        ).sum()

        match_percentage = (
            exact_matches / valid.sum()
        ) * 100

        mean_difference = difference.mean()

    verification_rows.append({
        "Feature": old_col,
        "Valid_Rows": int(valid.sum()),
        "Exact_Matches": int(exact_matches),
        "Match_Percentage": match_percentage,
        "Mean_Absolute_Difference": mean_difference
    })


verification_df = pd.DataFrame(
    verification_rows
)

print(verification_df.to_string(index=False))


verification_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_lag_rolling_verification.csv"
    ),
    index=False
)


# ============================================================
# 5. REPLACE FEATURES WITH VERIFIED FEATURES
# ============================================================

print("\nReplacing historical features with verified values...")

for old_col, verified_col in historical_features:

    df[old_col] = df[verified_col]


df = df.drop(
    columns=[
        "Verified_Lag_1",
        "Verified_Lag_2",
        "Verified_Lag_3",
        "Verified_Rolling_7",
        "Verified_Rolling_30"
    ]
)


# ============================================================
# 6. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 78)
print("HISTORICAL FEATURE MISSING VALUE CHECK")
print("=" * 78)

for col in [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]:

    print(
        f"{col}:",
        df[col].isna().sum()
    )


# Remove rows where historical features are unavailable.

historical_cols = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_7",
    "Rolling_30"
]

before_rows = len(df)

df = df.dropna(
    subset=historical_cols
).copy()

after_rows = len(df)

print(
    "\nRows removed:",
    before_rows - after_rows
)

print(
    "Rows available:",
    after_rows
)


# ============================================================
# 7. TARGET LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 78)
print("TARGET LEAKAGE CHECK")
print("=" * 78)

if "Price_Change" in df.columns:

    print(
        "Price_Change found."
    )

    print(
        "Removing Price_Change because it may contain today's target."
    )

    df = df.drop(
        columns=["Price_Change"]
    )

else:

    print(
        "✓ Price_Change is not present."
    )


# ============================================================
# 8. FEATURES
# ============================================================

target = "Modal_Price"

categorical_columns = [
    "State",
    "Commodity_Group",
    "Commodity",
    "District",
    "Festival"
]


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


xgb_features = rf_features.copy()


# ============================================================
# 9. TIME SPLIT
# ============================================================

print("\n" + "=" * 78)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 78)

unique_dates = sorted(
    df["Date"].dropna().unique()
)

split_index = int(
    len(unique_dates) * 0.8
)

split_date = unique_dates[split_index]

print(
    "Split date:",
    split_date
)

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

print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Testing rows :",
    len(test_df)
)


# ============================================================
# 10. ENCODING
# ============================================================

print("\n" + "=" * 78)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 78)

encoders = {}

for col in categorical_columns:

    le = LabelEncoder()

    # Fit ONLY using training data.

    train_values = train_df[col].astype(str)

    le.fit(train_values)

    train_df[col] = le.transform(
        train_values
    )

    # Handle possible unseen categories safely.

    mapping = {
        value: index
        for index, value in enumerate(le.classes_)
    }

    test_values = test_df[col].astype(str)

    test_df[col] = test_values.map(
        mapping
    ).fillna(-1).astype(int)

    encoders[col] = le

    print(
        f"{col}:",
        len(le.classes_),
        "categories"
    )


# ============================================================
# 11. BUILD X / y
# ============================================================

X_train = train_df[rf_features].copy()
X_test = test_df[rf_features].copy()

y_train = train_df[target].copy()
y_test = test_df[target].copy()


# ============================================================
# 12. RANDOM FOREST
# ============================================================

print("\n" + "=" * 78)
print("MODEL 1 - CORRECTED RANDOM FOREST")
print("=" * 78)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
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

print("\nCorrected Random Forest")

print(
    "MAE :",
    round(rf_mae, 4)
)

print(
    "RMSE:",
    round(rf_rmse, 4)
)

print(
    "R2  :",
    round(rf_r2, 4)
)


# ============================================================
# 13. XGBOOST CURRENT CONFIGURATION
# ============================================================

print("\n" + "=" * 78)
print("MODEL 2 - IMPROVED XGBOOST")
print("=" * 78)

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("Training Improved XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test
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

print("\nImproved XGBoost")

print(
    "MAE :",
    round(xgb_mae, 4)
)

print(
    "RMSE:",
    round(xgb_rmse, 4)
)

print(
    "R2  :",
    round(xgb_r2, 4)
)


# ============================================================
# 14. SECOND TUNED XGBOOST
# ============================================================

print("\n" + "=" * 78)
print("MODEL 3 - TUNED XGBOOST")
print("=" * 78)

xgb_tuned = XGBRegressor(
    n_estimators=800,
    learning_rate=0.025,
    max_depth=6,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.05,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("Training Tuned XGBoost...")

xgb_tuned.fit(
    X_train,
    y_train
)

tuned_pred = xgb_tuned.predict(
    X_test
)

tuned_mae = mean_absolute_error(
    y_test,
    tuned_pred
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_pred
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_pred
)

print("\nTuned XGBoost")

print(
    "MAE :",
    round(tuned_mae, 4)
)

print(
    "RMSE:",
    round(tuned_rmse, 4)
)

print(
    "R2  :",
    round(tuned_r2, 4)
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 78)
print("MODEL COMPARISON")
print("=" * 78)

comparison = pd.DataFrame([
    {
        "Model": "Corrected Random Forest",
        "MAE": rf_mae,
        "RMSE": rf_rmse,
        "R2": rf_r2
    },
    {
        "Model": "Improved XGBoost",
        "MAE": xgb_mae,
        "RMSE": xgb_rmse,
        "R2": xgb_r2
    },
    {
        "Model": "Tuned XGBoost",
        "MAE": tuned_mae,
        "RMSE": tuned_rmse,
        "R2": tuned_r2
    }
])

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

# Primary criterion: lowest MAE.
# Secondary criteria: RMSE and R2.

best_row = comparison.sort_values(
    by=["MAE", "RMSE", "R2"],
    ascending=[True, True, False]
).iloc[0]

best_name = best_row["Model"]

print("\n" + "=" * 78)
print("PRELIMINARY FINAL MODEL DECISION")
print("=" * 78)

print(
    "Best model:",
    best_name
)

print(
    "MAE:",
    round(best_row["MAE"], 4)
)

print(
    "RMSE:",
    round(best_row["RMSE"], 4)
)

print(
    "R2:",
    round(best_row["R2"], 4)
)


# ============================================================
# 17. SELECT MODEL OBJECT
# ============================================================

if best_name == "Corrected Random Forest":

    final_model = rf_model
    final_predictions = rf_pred

elif best_name == "Improved XGBoost":

    final_model = xgb_model
    final_predictions = xgb_pred

else:

    final_model = xgb_tuned
    final_predictions = tuned_pred


# ============================================================
# 18. OVERALL TEST PREDICTIONS
# ============================================================

overall_predictions = test_df[
    [
        "Date",
        "Commodity",
        "District",
        "Modal_Price"
    ]
].copy()

overall_predictions[
    "Predicted_Price"
] = final_predictions

overall_predictions[
    "Absolute_Error"
] = (
    overall_predictions["Modal_Price"]
    - overall_predictions["Predicted_Price"]
).abs()

overall_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 19. VALIDATION BY COMMODITY
# ============================================================

print("\n" + "=" * 78)
print("VALIDATION BY COMMODITY")
print("=" * 78)

commodity_results = []

for commodity in sorted(
    test_df["Commodity"].unique()
):

    mask = (
        test_df["Commodity"]
        == commodity
    )

    actual = y_test[mask]
    predicted = pd.Series(
        final_predictions,
        index=test_df.index
    )[mask]

    if len(actual) == 0:
        continue

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    ) if len(actual) > 1 else np.nan

    commodity_results.append({
        "Commodity": commodity,
        "Rows": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


commodity_df = pd.DataFrame(
    commodity_results
)

print(
    commodity_df.to_string(
        index=False
    )
)

commodity_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_by_commodity.csv"
    ),
    index=False
)


# ============================================================
# 20. VALIDATION BY DISTRICT
# ============================================================

print("\n" + "=" * 78)
print("VALIDATION BY DISTRICT")
print("=" * 78)

district_results = []

for district in sorted(
    test_df["District"].unique()
):

    mask = (
        test_df["District"]
        == district
    )

    actual = y_test[mask]

    predicted = pd.Series(
        final_predictions,
        index=test_df.index
    )[mask]

    if len(actual) == 0:
        continue

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    ) if len(actual) > 1 else np.nan

    district_results.append({
        "District": district,
        "Rows": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


district_df = pd.DataFrame(
    district_results
)

print(
    district_df.to_string(
        index=False
    )
)

district_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_by_district.csv"
    ),
    index=False
)


# ============================================================
# 21. WEAK COMMODITIES
# ============================================================

print("\n" + "=" * 78)
print("WEAK COMMODITIES")
print("=" * 78)

weak_commodities = commodity_df.sort_values(
    by="MAE",
    ascending=False
)

print(
    weak_commodities.head(3).to_string(
        index=False
    )
)


# ============================================================
# 22. WEAK DISTRICTS
# ============================================================

print("\n" + "=" * 78)
print("WEAK DISTRICTS")
print("=" * 78)

weak_districts = district_df.sort_values(
    by="MAE",
    ascending=False
)

print(
    weak_districts.head(5).to_string(
        index=False
    )
)


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 78)
print("FINAL MODEL FEATURE IMPORTANCE")
print("=" * 78)

if hasattr(
    final_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({
        "Feature": rf_features,
        "Importance": final_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print(
        importance_df.to_string(
            index=False
        )
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "phase6_final_feature_importance.csv"
        ),
        index=False
    )


# ============================================================
# 24. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_price_prediction_model.pkl"
)

final_encoder_path = os.path.join(
    MODEL_DIR,
    "final_label_encoders.pkl"
)

final_feature_path = os.path.join(
    MODEL_DIR,
    "final_feature_list.json"
)


joblib.dump(
    final_model,
    final_model_path
)

joblib.dump(
    encoders,
    final_encoder_path
)


with open(
    final_feature_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "features": rf_features,
            "target": target,
            "categorical_columns": categorical_columns,
            "model": best_name
        },
        f,
        indent=4
    )


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("PHASE 6 COMPLETED")
print("=" * 78)

print("\nFINAL MODEL:")
print(best_name)

print(
    "\nFinal MAE:",
    round(best_row["MAE"], 4)
)

print(
    "Final RMSE:",
    round(best_row["RMSE"], 4)
)

print(
    "Final R2:",
    round(best_row["R2"], 4)
)

print(
    "\nFinal feature count:",
    len(rf_features)
)

print("\nSaved files:")

print(
    "1.",
    final_model_path
)

print(
    "2.",
    final_encoder_path
)

print(
    "3.",
    final_feature_path
)

print(
    "4.",
    os.path.join(
        OUTPUT_DIR,
        "phase6_model_comparison.csv"
    )
)

print(
    "5.",
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_by_commodity.csv"
    )
)

print(
    "6.",
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_by_district.csv"
    )
)

print(
    "7.",
    os.path.join(
        OUTPUT_DIR,
        "phase6_final_test_predictions.csv"
    )
)

print("\n" + "=" * 78)