import os
import pandas as pd
import numpy as np


# ==============================================================
# FARMER PRICE INTELLIGENCE SYSTEM
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("FARMER PRICE INTELLIGENCE SYSTEM")
print("=" * 70)


# ==============================================================
# STEP 1 — LOAD CRISIS DATA
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 1 — LOAD AND VALIDATE CRISIS DATA")
print("=" * 70)


crisis_file = os.path.join(
    OUTPUT_DIR,
    "crisis_score_results.csv"
)


if not os.path.exists(crisis_file):
    raise FileNotFoundError(
        f"Crisis file not found:\n{crisis_file}"
    )


crisis_df = pd.read_csv(
    crisis_file
)


print("\nCrisis dataset loaded successfully.")
print("Shape:", crisis_df.shape)


# --------------------------------------------------------------
# REQUIRED COLUMNS
# --------------------------------------------------------------

required_columns = [
    "Date",
    "Commodity",
    "District",
    "Modal_Price",
    "Arrival",
    "Crisis_Score",
    "Crisis_Level"
]


print("\nRequired columns validation:")

for column in required_columns:

    print(
        f"{column:<20} ->",
        column in crisis_df.columns
    )

    if column not in crisis_df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )


# --------------------------------------------------------------
# DATE CONVERSION
# --------------------------------------------------------------

crisis_df["Date"] = pd.to_datetime(
    crisis_df["Date"],
    errors="coerce"
)


if crisis_df["Date"].isnull().any():

    raise ValueError(
        "Invalid dates found in crisis dataset."
    )


# --------------------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------------------

print(
    "Number of commodities:",
    crisis_df["Commodity"].nunique()
)

print(
    "Number of districts:",
    crisis_df["District"].nunique()
)

print(
    "Commodities:",
    crisis_df["Commodity"].unique()
)

print(
    "Districts:",
    crisis_df["District"].unique()
)

print(
    "Date range:",
    crisis_df["Date"].min(),
    "to",
    crisis_df["Date"].max()
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 1 COMPLETED")
print("=" * 70)


# ==============================================================
# LOAD KMEANS RESULTS
# ==============================================================

print("\nLoading KMeans district cluster data...")


kmeans_file = os.path.join(
    OUTPUT_DIR,
    "kmeans_mandi_clusters.csv"
)


if os.path.exists(kmeans_file):

    kmeans_df = pd.read_csv(
        kmeans_file
    )

    print(
        "KMeans cluster data loaded."
    )

    print(
        "Shape:",
        kmeans_df.shape
    )

else:

    kmeans_df = None

    print(
        "KMeans file not found."
    )


# ==============================================================
# LOAD OUTLIER RESULTS
# ==============================================================

print("\nLoading abnormal price/supply events...")


outlier_file = os.path.join(
    OUTPUT_DIR,
    "abnormal_price_supply_events.csv"
)


if os.path.exists(outlier_file):

    outlier_df = pd.read_csv(
        outlier_file
    )

    print(
        "Outlier data loaded."
    )

    print(
        "Shape:",
        outlier_df.shape
    )

else:

    outlier_df = None

    print(
        "Outlier file not found."
    )


# ==============================================================
# STEP 2 — LOAD XGBOOST FORECAST
# ==============================================================

print("\nLoading XGBoost 30-day price forecast...")


xgb_file = os.path.join(
    OUTPUT_DIR,
    "xgboost_30_day_price_forecast.csv"
)


if not os.path.exists(xgb_file):

    raise FileNotFoundError(
        f"XGBoost forecast file not found:\n{xgb_file}"
    )


xgb_df = pd.read_csv(
    xgb_file
)


print(
    "XGBoost forecast loaded successfully."
)

print(
    "Shape:",
    xgb_df.shape
)

print(
    "Columns:",
    list(xgb_df.columns)
)


# --------------------------------------------------------------
# VALIDATE FORECAST COLUMNS
# --------------------------------------------------------------

forecast_required = [
    "Date",
    "Commodity",
    "Predicted_Price"
]


for column in forecast_required:

    if column not in xgb_df.columns:

        raise ValueError(
            f"Forecast column missing: {column}"
        )


# --------------------------------------------------------------
# DATE CONVERSION
# --------------------------------------------------------------

xgb_df["Date"] = pd.to_datetime(
    xgb_df["Date"],
    errors="coerce"
)


if xgb_df["Date"].isnull().any():

    raise ValueError(
        "Invalid dates found in XGBoost forecast."
    )


# --------------------------------------------------------------
# SORT
# --------------------------------------------------------------

xgb_df = xgb_df.sort_values(
    [
        "Commodity",
        "Date"
    ]
).reset_index(
    drop=True
)


print(
    "Forecast commodities:",
    xgb_df["Commodity"].unique()
)

print(
    "Forecast date range:",
    xgb_df["Date"].min(),
    "to",
    xgb_df["Date"].max()
)


print("\nForecast records per commodity:")

print(
    xgb_df[
        "Commodity"
    ]
    .value_counts()
    .sort_index()
)


# ==============================================================
# STEP 3 — COMMODITY FORECAST INTELLIGENCE
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 2 — XGBOOST FORECAST INTELLIGENCE")
print("=" * 70)


forecast_summary = (
    xgb_df
    .groupby("Commodity")
    .agg(
        Forecast_Average_Price=(
            "Predicted_Price",
            "mean"
        ),

        Forecast_Min_Price=(
            "Predicted_Price",
            "min"
        ),

        Forecast_Max_Price=(
            "Predicted_Price",
            "max"
        ),

        Forecast_Std=(
            "Predicted_Price",
            "std"
        ),

        Forecast_First_Price=(
            "Predicted_Price",
            "first"
        ),

        Forecast_Last_Price=(
            "Predicted_Price",
            "last"
        )
    )
    .reset_index()
)


# --------------------------------------------------------------
# FORECAST CHANGE
# --------------------------------------------------------------

forecast_summary["Forecast_Change_pct"] = (

    (
        forecast_summary["Forecast_Last_Price"]
        -
        forecast_summary["Forecast_First_Price"]
    )

    /

    forecast_summary["Forecast_First_Price"]

) * 100


# --------------------------------------------------------------
# FORECAST RANGE
# --------------------------------------------------------------

forecast_summary["Forecast_Range_pct"] = (

    (
        forecast_summary["Forecast_Max_Price"]
        -
        forecast_summary["Forecast_Min_Price"]
    )

    /

    forecast_summary["Forecast_Min_Price"]

) * 100


# --------------------------------------------------------------
# ROUND
# --------------------------------------------------------------

forecast_numeric_columns = [

    "Forecast_Average_Price",
    "Forecast_Min_Price",
    "Forecast_Max_Price",
    "Forecast_Std",
    "Forecast_First_Price",
    "Forecast_Last_Price",
    "Forecast_Change_pct",
    "Forecast_Range_pct"
]


forecast_summary[
    forecast_numeric_columns
] = forecast_summary[
    forecast_numeric_columns
].round(2)


print("\n30-DAY FORECAST SUMMARY")

print(
    forecast_summary
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 2 COMPLETED")
print("=" * 70)


# ==============================================================
# STEP 4 — HISTORICAL RISK INTELLIGENCE
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 3 — HISTORICAL RISK INTELLIGENCE")
print("=" * 70)


# --------------------------------------------------------------
# CRISIS SUMMARY
# --------------------------------------------------------------

crisis_summary = (
    crisis_df
    .groupby("Commodity")
    .agg(

        Average_Crisis_Score=(
            "Crisis_Score",
            "mean"
        ),

        Maximum_Crisis_Score=(
            "Crisis_Score",
            "max"
        ),

        High_Critical_Events=(
            "Crisis_Level",
            lambda x:
            x.isin(
                [
                    "High",
                    "Critical"
                ]
            ).sum()
        ),

        Total_Crisis_Records=(
            "Crisis_Level",
            "count"
        )
    )
    .reset_index()
)


# --------------------------------------------------------------
# OUTLIER SUMMARY
# --------------------------------------------------------------

if outlier_df is not None:

    outlier_summary = (

        outlier_df
        .groupby("Commodity")
        .size()
        .reset_index(
            name="Abnormal_Events"
        )

    )

else:

    outlier_summary = pd.DataFrame(
        columns=[
            "Commodity",
            "Abnormal_Events"
        ]
    )


# --------------------------------------------------------------
# MERGE
# --------------------------------------------------------------

risk_summary = pd.merge(
    crisis_summary,
    outlier_summary,
    on="Commodity",
    how="left"
)


risk_summary["Abnormal_Events"] = (
    risk_summary["Abnormal_Events"]
    .fillna(0)
)


# --------------------------------------------------------------
# HIGH / CRITICAL %
# --------------------------------------------------------------

risk_summary["High_Critical_%"] = (

    risk_summary["High_Critical_Events"]
    /
    risk_summary["Total_Crisis_Records"]

) * 100


# --------------------------------------------------------------
# ROUND
# --------------------------------------------------------------

risk_numeric_columns = [

    "Average_Crisis_Score",
    "Maximum_Crisis_Score",
    "High_Critical_%",
    "Abnormal_Events"
]


risk_summary[
    risk_numeric_columns
] = risk_summary[
    risk_numeric_columns
].round(2)


print("\nHISTORICAL RISK SUMMARY")

print(
    risk_summary
    .sort_values(
        "Average_Crisis_Score",
        ascending=False
    )
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 3 COMPLETED")
print("=" * 70)


# ==============================================================
# STEP 5 — FINAL COMMODITY FARMER INTELLIGENCE
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 4 — FINAL FARMER INTELLIGENCE SCORE")
print("=" * 70)


# --------------------------------------------------------------
# MERGE FORECAST + HISTORICAL RISK
# --------------------------------------------------------------

final_intelligence = pd.merge(

    forecast_summary,

    risk_summary,

    on="Commodity",

    how="left"
)


# --------------------------------------------------------------
# CRISIS RISK SCORE
# --------------------------------------------------------------

final_intelligence["Crisis_Risk_Score"] = (

    final_intelligence[
        "Average_Crisis_Score"
    ]
    .clip(0, 100)

)


# --------------------------------------------------------------
# FORECAST VOLATILITY SCORE
# --------------------------------------------------------------

max_volatility = (
    final_intelligence[
        "Forecast_Std"
    ].max()
)


if max_volatility > 0:

    final_intelligence[
        "Forecast_Volatility_Score"
    ] = (

        final_intelligence[
            "Forecast_Std"
        ]

        /

        max_volatility

    ) * 100

else:

    final_intelligence[
        "Forecast_Volatility_Score"
    ] = 0


# --------------------------------------------------------------
# CRISIS EVENT SCORE
# --------------------------------------------------------------

max_events = (
    final_intelligence[
        "High_Critical_Events"
    ].max()
)


if max_events > 0:

    final_intelligence[
        "Crisis_Event_Score"
    ] = (

        final_intelligence[
            "High_Critical_Events"
        ]

        /

        max_events

    ) * 100

else:

    final_intelligence[
        "Crisis_Event_Score"
    ] = 0


# --------------------------------------------------------------
# FINAL SCORE
# --------------------------------------------------------------

final_intelligence[
    "Farmer_Intelligence_Score"
] = (

    final_intelligence[
        "Crisis_Risk_Score"
    ] * 0.40

    +

    final_intelligence[
        "Forecast_Volatility_Score"
    ] * 0.30

    +

    final_intelligence[
        "Crisis_Event_Score"
    ] * 0.30

)


final_intelligence[
    "Farmer_Intelligence_Score"
] = (

    final_intelligence[
        "Farmer_Intelligence_Score"
    ]

    .clip(0, 100)

    .round(2)

)


# --------------------------------------------------------------
# RISK LEVEL
# --------------------------------------------------------------

def assign_risk(score):

    if score >= 70:
        return "Critical"

    elif score >= 50:
        return "High"

    elif score >= 25:
        return "Moderate"

    else:
        return "Low"


final_intelligence[
    "Final_Risk_Level"
] = (

    final_intelligence[
        "Farmer_Intelligence_Score"
    ]
    .apply(assign_risk)

)


print("\nFINAL FARMER INTELLIGENCE")

print(

    final_intelligence[
        [
            "Commodity",
            "Forecast_Average_Price",
            "Forecast_Change_pct",
            "Average_Crisis_Score",
            "High_Critical_Events",
            "Forecast_Volatility_Score",
            "Farmer_Intelligence_Score",
            "Final_Risk_Level"
        ]
    ]
    .sort_values(
        "Farmer_Intelligence_Score",
        ascending=False
    )
    .to_string(index=False)

)


print("\n" + "=" * 70)
print("INTEGRATION STEP 4 COMPLETED")
print("=" * 70)


# ==============================================================
# STEP 6 — DISTRICT / MANDI INTELLIGENCE
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 5 — DISTRICT / MANDI INTELLIGENCE")
print("=" * 70)


if kmeans_df is None:

    raise FileNotFoundError(
        "KMeans district cluster file is not available."
    )


print("\nKMeans data loaded.")
print(
    "KMeans columns:",
    list(kmeans_df.columns)
)


# --------------------------------------------------------------
# DISTRICT CRISIS INTELLIGENCE
# --------------------------------------------------------------

district_crisis = (

    crisis_df
    .groupby("District")
    .agg(

        Average_Crisis_Score=(
            "Crisis_Score",
            "mean"
        ),

        Maximum_Crisis_Score=(
            "Crisis_Score",
            "max"
        ),

        High_Critical_Events=(
            "Crisis_Level",
            lambda x:
            x.isin(
                [
                    "High",
                    "Critical"
                ]
            ).sum()
        ),

        Total_Records=(
            "Crisis_Score",
            "count"
        ),

        Average_Price=(
            "Modal_Price",
            "mean"
        ),

        Average_Arrival=(
            "Arrival",
            "mean"
        )
    )
    .reset_index()

)


# --------------------------------------------------------------
# HIGH / CRITICAL %
# --------------------------------------------------------------

district_crisis[
    "High_Critical_%"
] = (

    district_crisis[
        "High_Critical_Events"
    ]

    /

    district_crisis[
        "Total_Records"
    ]

) * 100


# --------------------------------------------------------------
# CHECK DISTRICT PRICE VARIATION
# --------------------------------------------------------------

if (
    district_crisis["Average_Price"].nunique()
    == 1
):

    print(
        "\nWARNING:"
        "\nAverage_Price is identical for all districts."
        "\nThis indicates the source crisis dataset may contain"
        "\nidentical price values across districts."
    )

else:

    print(
        "\nDistrict average prices vary correctly."
    )


if (
    district_crisis["Average_Arrival"].nunique()
    == 1
):

    print(
        "\nWARNING:"
        "\nAverage_Arrival is identical for all districts."
        "\nPlease verify the source arrival data."
    )

else:

    print(
        "\nDistrict average arrivals vary correctly."
    )


# --------------------------------------------------------------
# PREPARE KMEANS
# --------------------------------------------------------------

kmeans_for_merge = kmeans_df.drop(

    columns=[
        "Average_Price",
        "Average_Arrival"
    ],

    errors="ignore"

)


# --------------------------------------------------------------
# MERGE
# --------------------------------------------------------------

district_intelligence = pd.merge(

    kmeans_for_merge,

    district_crisis,

    on="District",

    how="left"

)


# --------------------------------------------------------------
# DISTRICT RISK SCORE
# --------------------------------------------------------------

# IMPORTANT:
# We DO NOT min-max normalize here.
# The previous version converted the highest district
# automatically into 100 / Critical.

district_intelligence[
    "District_Risk_Score"
] = (

    district_intelligence[
        "Average_Crisis_Score"
    ].clip(0, 100) * 0.60

    +

    district_intelligence[
        "High_Critical_%"
    ].clip(0, 100) * 0.40

)


district_intelligence[
    "District_Risk_Score"
] = (

    district_intelligence[
        "District_Risk_Score"
    ]
    .clip(0, 100)
    .round(2)

)


# --------------------------------------------------------------
# DISTRICT RISK LEVEL
# --------------------------------------------------------------

def get_district_risk(score):

    if score >= 70:
        return "Critical"

    elif score >= 50:
        return "High"

    elif score >= 25:
        return "Moderate"

    else:
        return "Low"


district_intelligence[
    "District_Risk_Level"
] = (

    district_intelligence[
        "District_Risk_Score"
    ]
    .apply(get_district_risk)

)


# --------------------------------------------------------------
# ROUND
# --------------------------------------------------------------

district_numeric_columns = [

    "Average_Crisis_Score",
    "Maximum_Crisis_Score",
    "High_Critical_Events",
    "High_Critical_%",
    "Average_Price",
    "Average_Arrival",
    "District_Risk_Score"

]


district_intelligence[
    district_numeric_columns
] = (

    district_intelligence[
        district_numeric_columns
    ].round(2)

)


# --------------------------------------------------------------
# DISPLAY
# --------------------------------------------------------------

district_display_columns = [

    "District",
    "Cluster",
    "Average_Crisis_Score",
    "Maximum_Crisis_Score",
    "High_Critical_Events",
    "High_Critical_%",
    "Average_Price",
    "Average_Arrival",
    "District_Risk_Score",
    "District_Risk_Level"

]


print("\nFINAL DISTRICT / MANDI INTELLIGENCE")

print(

    district_intelligence[
        district_display_columns
    ]
    .sort_values(
        "District_Risk_Score",
        ascending=False
    )
    .to_string(index=False)

)


# --------------------------------------------------------------
# SAVE DISTRICT DATA
# --------------------------------------------------------------

district_output_file = os.path.join(

    OUTPUT_DIR,

    "district_mandi_intelligence.csv"

)


district_intelligence.to_csv(

    district_output_file,

    index=False

)


print(
    "\nDistrict intelligence saved:"
)

print(
    district_output_file
)


# ==============================================================
# CLUSTER SUMMARY
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 6 — CLUSTER SUMMARY")
print("=" * 70)


cluster_summary = (

    district_intelligence
    .groupby("Cluster")
    .agg(

        District_Count=(
            "District",
            "count"
        ),

        Average_Crisis_Score=(
            "Average_Crisis_Score",
            "mean"
        ),

        Average_Risk_Score=(
            "District_Risk_Score",
            "mean"
        ),

        High_Critical_Events=(
            "High_Critical_Events",
            "sum"
        )

    )
    .reset_index()

)


cluster_summary = (
    cluster_summary
    .round(2)
)


print(
    cluster_summary
    .to_string(index=False)
)


cluster_output_file = os.path.join(

    OUTPUT_DIR,

    "cluster_summary.csv"

)


cluster_summary.to_csv(

    cluster_output_file,

    index=False

)


print(
    "\nCluster summary saved:"
)

print(
    cluster_output_file
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 6 COMPLETED")
print("=" * 70)


# ==============================================================
# STEP 7 — FARMER RECOMMENDATION ENGINE
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 7 — FARMER RECOMMENDATION ENGINE")
print("=" * 70)


# --------------------------------------------------------------
# COMMODITY × DISTRICT
# --------------------------------------------------------------

recommendation_df = (

    final_intelligence
    .merge(
        district_intelligence[
            [
                "District",
                "Cluster",
                "District_Risk_Score",
                "District_Risk_Level"
            ]
        ],
        how="cross"
    )

)


print(
    "\nCommodity × District combinations:",
    len(recommendation_df)
)


# --------------------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------------------

def generate_recommendation(row):

    risk = row[
        "Final_Risk_Level"
    ]

    district_risk = row[
        "District_Risk_Level"
    ]

    price_change = row[
        "Forecast_Change_pct"
    ]

    volatility = row[
        "Forecast_Volatility_Score"
    ]

    crisis_score = row[
        "Average_Crisis_Score"
    ]


    # ----------------------------------------------------------
    # CRITICAL DISTRICT
    # ----------------------------------------------------------

    if district_risk == "Critical":

        return (
            "Critical district risk - monitor the market closely, "
            "avoid distress selling, and compare nearby mandi "
            "prices before selling."
        )


    # ----------------------------------------------------------
    # HIGH OVERALL RISK
    # ----------------------------------------------------------

    if (
        risk == "Critical"
        or risk == "High"
    ):

        return (
            "High commodity risk - avoid selling the entire stock "
            "at once and monitor price and supply conditions."
        )


    # ----------------------------------------------------------
    # HIGH VOLATILITY
    # ----------------------------------------------------------

    if volatility >= 70:

        return (
            "High forecast volatility - avoid making a single "
            "large selling decision and monitor price movement."
        )


    # ----------------------------------------------------------
    # FALLING PRICE
    # ----------------------------------------------------------

    if price_change <= -10:

        return (
            "Price decline expected - consider selling gradually "
            "while monitoring market conditions."
        )


    # ----------------------------------------------------------
    # RISING PRICE
    # ----------------------------------------------------------

    if price_change >= 10:

        return (
            "Positive price trend expected - consider holding "
            "part of the stock if storage is available."
        )


    # ----------------------------------------------------------
    # MODERATE CRISIS
    # ----------------------------------------------------------

    if crisis_score >= 25:

        return (
            "Moderate historical risk - monitor mandi prices, "
            "supply conditions, and weather before selling."
        )


    # ----------------------------------------------------------
    # DEFAULT
    # ----------------------------------------------------------

    return (
        "Stable conditions - monitor mandi prices, supply "
        "conditions, and weather before making a decision."
    )


# --------------------------------------------------------------
# APPLY
# --------------------------------------------------------------

recommendation_df[
    "Recommendation"
] = (

    recommendation_df
    .apply(
        generate_recommendation,
        axis=1
    )

)


# ==============================================================
# ACTION CATEGORY
# ==============================================================

def generate_action(row):

    recommendation = row[
        "Recommendation"
    ].lower()

    district_risk = row[
        "District_Risk_Level"
    ]


    if district_risk == "Critical":

        return "AVOID DISTRESS SELLING"


    if "decline" in recommendation:

        return "SELL GRADUALLY"


    if "holding" in recommendation:

        return "CONSIDER HOLDING"


    if "high commodity risk" in recommendation:

        return "MONITOR MARKET"


    return "MONITOR MARKET"


recommendation_df[
    "Recommended_Action"
] = (

    recommendation_df
    .apply(
        generate_action,
        axis=1
    )

)


# ==============================================================
# FINAL RECOMMENDATION COLUMNS
# ==============================================================

recommendation_output = (

    recommendation_df[
        [
            "Commodity",
            "Forecast_Average_Price",
            "Forecast_Min_Price",
            "Forecast_Max_Price",
            "Forecast_Change_pct",
            "Average_Crisis_Score",
            "High_Critical_Events",
            "Forecast_Volatility_Score",
            "Farmer_Intelligence_Score",
            "Final_Risk_Level",
            "District",
            "Cluster",
            "District_Risk_Score",
            "District_Risk_Level",
            "Recommendation",
            "Recommended_Action"
        ]
    ]
    .copy()

)


# --------------------------------------------------------------
# ROUND
# --------------------------------------------------------------

recommendation_numeric_columns = [

    "Forecast_Average_Price",
    "Forecast_Min_Price",
    "Forecast_Max_Price",
    "Forecast_Change_pct",
    "Average_Crisis_Score",
    "Forecast_Volatility_Score",
    "Farmer_Intelligence_Score",
    "District_Risk_Score"

]


recommendation_output[
    recommendation_numeric_columns
] = (

    recommendation_output[
        recommendation_numeric_columns
    ].round(2)

)


# --------------------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------------------

recommendation_output = (

    recommendation_output
    .drop_duplicates(
        subset=[
            "Commodity",
            "District"
        ]
    )

)


# --------------------------------------------------------------
# SAVE
# --------------------------------------------------------------

recommendation_file = os.path.join(

    OUTPUT_DIR,

    "farmer_recommendations.csv"

)


recommendation_output.to_csv(

    recommendation_file,

    index=False

)


print("\nFINAL FARMER RECOMMENDATIONS")

print(

    recommendation_output
    .sort_values(
        "District_Risk_Score",
        ascending=False
    )
    .head(20)
    .to_string(index=False)

)


print(
    "\nTotal recommendation records:",
    len(recommendation_output)
)


print(
    "\nRecommendation output saved:"
)

print(
    recommendation_file
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 7 COMPLETED")
print("=" * 70)

# ==============================================================
# BEST FORECASTED DISTRICT
# ==============================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST FORECASTED DISTRICT BY COMMODITY"
)

print(
    "=" * 70
)


# ==============================================================
# BEST FORECASTED DISTRICT BY COMMODITY
# ==============================================================

print(
    "\n" + "=" * 70
)

print(
    "BEST FORECASTED DISTRICT BY COMMODITY"
)

print(
    "=" * 70
)


# --------------------------------------------------------------
# CALCULATE DIRECTLY FROM DISTRICT-LEVEL FORECAST DATA
# --------------------------------------------------------------

district_forecast_comparison = (

    xgb_df

    .groupby(
        [
            "Commodity",
            "District"
        ]
    )

    ["Predicted_Price"]

    .mean()

    .reset_index(
        name="Best_District_Forecast_Average"
    )
)


# --------------------------------------------------------------
# FIND HIGHEST FORECAST DISTRICT FOR EACH COMMODITY
# --------------------------------------------------------------

best_district_by_commodity = (

    district_forecast_comparison

    .sort_values(
        [
            "Commodity",
            "Best_District_Forecast_Average"
        ],
        ascending=[
            True,
            False
        ]
    )

    .drop_duplicates(
        subset=[
            "Commodity"
        ]
    )

    .rename(
        columns={
            "District":
                "Best_Forecasted_District"
        }
    )

    .reset_index(
        drop=True
    )
)


# --------------------------------------------------------------
# DISPLAY
# --------------------------------------------------------------

print(
    best_district_by_commodity
    .to_string(
        index=False
    )
)


# --------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------

if (
    len(best_district_by_commodity)
    !=
    xgb_df["Commodity"].nunique()
):

    raise ValueError(
        "Best forecasted district calculation is incomplete."
    )# ==============================================================
# STEP 8 — FINAL MASTER DATASET
# ==============================================================

print("\n" + "=" * 70)
print("INTEGRATION STEP 8 — FINAL MASTER INTELLIGENCE DATASET")
print("=" * 70)


master_df = recommendation_output.copy()

# --------------------------------------------------------------
# ADD BEST FORECASTED DISTRICT
# --------------------------------------------------------------

master_df = pd.merge(

    master_df,

    best_district_by_commodity[
        [
            "Commodity",
            "Best_Forecasted_District",
            "Best_District_Forecast_Average"
        ]
    ],

    on="Commodity",

    how="left"
)


# --------------------------------------------------------------
# MASTER COLUMNS
# --------------------------------------------------------------

master_columns = [

    "Commodity",

    "Forecast_Average_Price",
    "Forecast_Min_Price",
    "Forecast_Max_Price",
    "Forecast_Change_pct",
    "Forecast_Volatility_Score",

    "Average_Crisis_Score",
    "High_Critical_Events",

    "Farmer_Intelligence_Score",
    "Final_Risk_Level",

    "District",
    "Cluster",
    "District_Risk_Score",
    "District_Risk_Level",

   "Best_Forecasted_District",
   "Best_District_Forecast_Average",

   "Recommendation",
   "Recommended_Action"
]


master_df = master_df[
    master_columns
].copy()


# --------------------------------------------------------------
# SORT
# --------------------------------------------------------------

master_df = (

    master_df
    .sort_values(
        [
            "Commodity",
            "District"
        ]
    )
    .reset_index(drop=True)

)


# --------------------------------------------------------------
# FINAL VALIDATION
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MASTER DATASET VALIDATION")
print("=" * 70)


print(
    "\nDataset shape:",
    master_df.shape
)


print(
    "Number of commodities:",
    master_df["Commodity"].nunique()
)


print(
    "Number of districts:",
    master_df["District"].nunique()
)


expected_combinations = (

    master_df["Commodity"].nunique()
    *
    master_df["District"].nunique()

)


actual_combinations = (

    master_df[
        [
            "Commodity",
            "District"
        ]
    ]
    .drop_duplicates()
    .shape[0]

)


print(
    "Expected commodity-district combinations:",
    expected_combinations
)


print(
    "Actual commodity-district combinations:",
    actual_combinations
)


# --------------------------------------------------------------
# MISSING VALUES
# --------------------------------------------------------------

print("\nMissing values:")

print(
    master_df.isnull().sum()
)


# --------------------------------------------------------------
# RISK DISTRIBUTION
# --------------------------------------------------------------

print("\nFinal Risk Level Distribution:")

print(
    master_df[
        "Final_Risk_Level"
    ]
    .value_counts()
)


print("\nDistrict Risk Level Distribution:")

print(
    master_df[
        "District_Risk_Level"
    ]
    .value_counts()
)


print("\nRecommended Action Distribution:")

print(
    master_df[
        "Recommended_Action"
    ]
    .value_counts()
)


# --------------------------------------------------------------
# SAVE MASTER DATASET
# --------------------------------------------------------------

master_output_file = os.path.join(

    OUTPUT_DIR,

    "final_farmer_intelligence.csv"

)


master_df.to_csv(

    master_output_file,

    index=False

)


print("\n" + "=" * 70)
print("FINAL MASTER DATASET SAVED")
print("=" * 70)


print(
    master_output_file
)


print(
    "\nFinal records:",
    len(master_df)
)


print("\n" + "=" * 70)
print("INTEGRATION STEP 8 COMPLETED")
print("=" * 70)


print("\n" + "=" * 70)
print("FARMER PRICE INTELLIGENCE SYSTEM COMPLETED SUCCESSFULLY")
print("=" * 70)