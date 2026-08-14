import os
import json
import joblib
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
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
    "final_dataset_phase5.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_price_prediction_model.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_label_encoders.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_feature_list.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "xgboost_30_day_price_forecast.csv"
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("CURRENT 30-DAY FORECAST GENERATOR")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Date",
        "Commodity",
        "District",
        "Modal_Price"
    ]
)

df = df.sort_values(
    [
        "Commodity",
        "District",
        "Date"
    ]
).reset_index(drop=True)


print(
    "Dataset shape:",
    df.shape
)

print(
    "Historical range:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


# ==========================================================
# LOAD FINAL MODEL
# ==========================================================

model = joblib.load(
    MODEL_PATH
)

encoders = joblib.load(
    ENCODER_PATH
)

with open(
    FEATURE_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)


FEATURES = feature_info["features"]


print(
    "\nModel:",
    type(model).__name__
)

print(
    "Number of features:",
    len(FEATURES)
)

print(
    "Features:",
    FEATURES
)


# ==========================================================
# ENCODING HELPER
# ==========================================================

def encode_value(
    column,
    value
):

    encoder = encoders[column]

    return encoder.transform(
        [value]
    )[0]


# ==========================================================
# FORECAST START DATE
# ==========================================================

# The project is being run on 13-Aug-2026.
# Therefore forecast begins on 14-Aug-2026.

forecast_start = pd.Timestamp(
    "2026-08-14"
)

forecast_dates = pd.date_range(
    start=forecast_start,
    periods=30,
    freq="D"
)


print(
    "\nForecast range:",
    forecast_dates.min().date(),
    "to",
    forecast_dates.max().date()
)


# ==========================================================
# FORECAST
# ==========================================================

forecast_results = []


grouped = df.groupby(
    [
        "Commodity",
        "District"
    ],
    sort=True
)


for (
    commodity,
    district
), group in grouped:

    print(
        f"\nForecasting {commodity} - {district}"
    )

    group = (
        group
        .sort_values("Date")
        .reset_index(drop=True)
    )


    # ------------------------------------------------------
    # Need at least 30 historical prices
    # ------------------------------------------------------

    if len(group) < 30:

        print(
            "Skipped: less than 30 records."
        )

        continue


    # ------------------------------------------------------
    # Latest known row
    # ------------------------------------------------------

    latest = group.iloc[-1]


    # ------------------------------------------------------
    # Historical price history
    # ------------------------------------------------------

    history = list(
        group["Modal_Price"]
        .astype(float)
        .values
    )


    # ------------------------------------------------------
    # Hold non-price variables at latest observed values
    # ------------------------------------------------------

    state = latest["State"]

    commodity_group = latest[
        "Commodity_Group"
    ]

    arrival = float(
        latest["Arrival"]
    )

    max_temp = float(
        latest["Max_Temp"]
    )

    min_temp = float(
        latest["Min_Temp"]
    )

    rainfall = float(
        latest["Rainfall_mm"]
    )

    windspeed = float(
        latest["WindSpeed"]
    )

    festival = latest["Festival"]

    holiday = int(
        latest["Holiday"]
    )


    # ------------------------------------------------------
    # Encode categorical values
    # ------------------------------------------------------

    state_encoded = encode_value(
        "State",
        state
    )

    commodity_group_encoded = (
        encode_value(
            "Commodity_Group",
            commodity_group
        )
    )

    commodity_encoded = encode_value(
        "Commodity",
        commodity
    )

    district_encoded = encode_value(
        "District",
        district
    )

    festival_encoded = encode_value(
        "Festival",
        festival
    )


    # ------------------------------------------------------
    # Recursive forecasting
    # ------------------------------------------------------

    for future_date in forecast_dates:

        # ------------------------------
        # Lag features
        # ------------------------------

        lag1 = history[-1]

        lag2 = history[-2]

        lag3 = history[-3]


        # ------------------------------
        # Rolling features
        # ------------------------------

        rolling7 = np.mean(
            history[-7:]
        )

        rolling30 = np.mean(
            history[-30:]
        )


        # ------------------------------
        # Calendar features
        # ------------------------------

        month = future_date.month

        year = future_date.year

        day = future_date.day

        dayofweek = (
            future_date.dayofweek
        )

        weekend = (
            1
            if dayofweek >= 5
            else 0
        )

        quarter = future_date.quarter


        # ------------------------------
        # Model row
        # ------------------------------

        row = {

            "State":
                state_encoded,

            "Commodity_Group":
                commodity_group_encoded,

            "Commodity":
                commodity_encoded,

            "Arrival":
                arrival,

            "District":
                district_encoded,

            "Max_Temp":
                max_temp,

            "Min_Temp":
                min_temp,

            "Rainfall_mm":
                rainfall,

            "WindSpeed":
                windspeed,

            "Festival":
                festival_encoded,

            "Holiday":
                holiday,

            "Month":
                month,

            "Year":
                year,

            "Day":
                day,

            "Weekend":
                weekend,

            "DayOfWeek":
                dayofweek,

            "Quarter":
                quarter,

            "Lag_1":
                lag1,

            "Lag_2":
                lag2,

            "Lag_3":
                lag3,

            "Rolling_7":
                rolling7,

            "Rolling_30":
                rolling30
        }


        X_future = pd.DataFrame(
            [row]
        )


        X_future = X_future[
            FEATURES
        ]


        # ------------------------------
        # Predict
        # ------------------------------

        prediction = model.predict(
            X_future
        )[0]


        prediction = max(
            0.0,
            float(prediction)
        )


        # ------------------------------
        # Store result
        # ------------------------------

        forecast_results.append({

            "Date":
                future_date.strftime(
                    "%Y-%m-%d"
                ),

            "Commodity":
                commodity,

            "District":
                district,

            "Predicted_Price":
                round(
                    prediction,
                    2
                )
        })


        # ------------------------------
        # Recursive update
        # ------------------------------

        history.append(
            prediction
        )


# ==========================================================
# SAVE
# ==========================================================

forecast_df = pd.DataFrame(
    forecast_results
)


forecast_df = forecast_df.sort_values(
    [
        "Commodity",
        "District",
        "Date"
    ]
).reset_index(
    drop=True
)


forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 70)
print("FORECAST COMPLETED")
print("=" * 70)

print(
    "Output shape:",
    forecast_df.shape
)

print(
    "Output file:",
    OUTPUT_PATH
)

print(
    "\nForecast dates:"
)

print(
    forecast_df["Date"].min(),
    "to",
    forecast_df["Date"].max()
)

print(
    "\nRecords per commodity:"
)

print(
    forecast_df["Commodity"]
    .value_counts()
    .sort_index()
)

print(
    "\nRecords per commodity-district:"
)

print(
    forecast_df
    .groupby(
        [
            "Commodity",
            "District"
        ]
    )
    .size()
    .describe()
)

print(
    "\nFirst 20 rows:"
)

print(
    forecast_df.head(20).to_string(
        index=False
    )
)

print(
    "\n30-day forecast saved successfully."
)