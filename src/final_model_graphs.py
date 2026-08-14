import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("FINAL MODEL GRAPH GENERATION")
print("=" * 70)


# ============================================================
# 1. MODEL COMPARISON GRAPH
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "phase6_model_comparison.csv"
)

df_model = pd.read_csv(model_file)

print("\nModel comparison:")
print(df_model)


plt.figure(figsize=(10, 6))

plt.bar(
    df_model["Model"],
    df_model["R2"]
)

plt.ylabel("R² Score")
plt.xlabel("Model")
plt.title("Final Model Comparison - R² Score")

plt.xticks(rotation=15)
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_r2.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# 2. MAE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df_model["Model"],
    df_model["MAE"]
)

plt.ylabel("MAE")
plt.xlabel("Model")
plt.title("Final Model Comparison - MAE")

plt.xticks(rotation=15)
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_mae.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# 3. RMSE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df_model["Model"],
    df_model["RMSE"]
)

plt.ylabel("RMSE")
plt.xlabel("Model")
plt.title("Final Model Comparison - RMSE")

plt.xticks(rotation=15)
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_rmse.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# 4. FEATURE IMPORTANCE
# ============================================================

feature_file = os.path.join(
    OUTPUT_DIR,
    "phase6_final_feature_importance.csv"
)

df_feature = pd.read_csv(feature_file)

print("\nFeature importance:")
print(df_feature)


df_feature = df_feature.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    df_feature["Feature"],
    df_feature["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Tuned XGBoost - Feature Importance")

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "final_feature_importance.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# 5. COMMODITY PERFORMANCE
# ============================================================

commodity_file = os.path.join(
    OUTPUT_DIR,
    "phase6_final_by_commodity.csv"
)

df_commodity = pd.read_csv(commodity_file)

print("\nCommodity validation:")
print(df_commodity)


plt.figure(figsize=(10, 6))

plt.bar(
    df_commodity["Commodity"].astype(str),
    df_commodity["MAE"]
)

plt.xlabel("Commodity")
plt.ylabel("MAE")
plt.title("Model Performance by Commodity")

plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "commodity_mae_comparison.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# 6. DISTRICT PERFORMANCE
# ============================================================

district_file = os.path.join(
    OUTPUT_DIR,
    "phase6_final_by_district.csv"
)

df_district = pd.read_csv(district_file)

print("\nDistrict validation:")
print(df_district)


plt.figure(figsize=(10, 6))

plt.bar(
    df_district["District"].astype(str),
    df_district["MAE"]
)

plt.xlabel("District")
plt.ylabel("MAE")
plt.title("Model Performance by District")

plt.xticks(rotation=45)
plt.tight_layout()

path = os.path.join(
    OUTPUT_DIR,
    "district_mae_comparison.png"
)

plt.savefig(path, dpi=300)
plt.close()

print("Created:", path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ALL FINAL MODEL GRAPHS CREATED SUCCESSFULLY")
print("=" * 70)