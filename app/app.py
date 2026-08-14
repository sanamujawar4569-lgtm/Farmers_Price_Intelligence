from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import json
from datetime import datetime


# ==============================================================
# FLASK APPLICATION
# ==============================================================

app = Flask(__name__)


# ==============================================================
# PROJECT DIRECTORIES
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


# ==============================================================
# FILE PATHS
# ==============================================================

HISTORICAL_DATA_PATH = os.path.join(
    DATA_DIR,
    "final_dataset_phase5.csv"
)

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

INTELLIGENCE_PATH = os.path.join(
    OUTPUT_DIR,
    "final_farmer_intelligence.csv"
)

RECOMMENDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "farmer_recommendations.csv"
)

DISTRICT_INTELLIGENCE_PATH = os.path.join(
    OUTPUT_DIR,
    "district_mandi_intelligence.csv"
)

FORECAST_PATH = os.path.join(
    OUTPUT_DIR,
    "xgboost_30_day_price_forecast.csv"
)


# ==============================================================
# STARTUP MESSAGE
# ==============================================================

print("=" * 70)
print("FARMER PRICE INTELLIGENCE - FLASK APPLICATION")
print("=" * 70)


# ==============================================================
# CHECK REQUIRED FILES
# ==============================================================

required_files = {

    "Historical Dataset":
        HISTORICAL_DATA_PATH,

    "Final Model":
        FINAL_MODEL_PATH,

    "Final Encoders":
        FINAL_ENCODER_PATH,

    "Feature List":
        FINAL_FEATURE_PATH,

    "Farmer Intelligence":
        INTELLIGENCE_PATH,

    "Farmer Recommendations":
        RECOMMENDATION_PATH,

    "District Intelligence":
        DISTRICT_INTELLIGENCE_PATH,

    "30-Day Forecast":
        FORECAST_PATH

}


for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


# ==============================================================
# LOAD HISTORICAL DATA
# ==============================================================

historical_df = pd.read_csv(
    HISTORICAL_DATA_PATH
)

historical_df["Date"] = pd.to_datetime(
    historical_df["Date"],
    errors="coerce"
)

historical_df = historical_df.dropna(
    subset=[
        "Date",
        "Commodity",
        "District",
        "Modal_Price"
    ]
)

historical_df = historical_df.sort_values(
    [
        "Commodity",
        "District",
        "Date"
    ]
).reset_index(
    drop=True
)

historical_df["Modal_Price"] = pd.to_numeric(
    historical_df["Modal_Price"],
    errors="coerce"
)

historical_df = historical_df.dropna(
    subset=["Modal_Price"]
)


print(
    "Historical dataset loaded:",
    historical_df.shape
)

print(
    "Historical date range:",
    historical_df["Date"].min().date(),
    "to",
    historical_df["Date"].max().date()
)


# ==============================================================
# LOAD MODEL
# ==============================================================

model = joblib.load(
    FINAL_MODEL_PATH
)

encoders = joblib.load(
    FINAL_ENCODER_PATH
)


with open(
    FINAL_FEATURE_PATH,
    "r",
    encoding="utf-8"
) as f:

    feature_info = json.load(f)


FEATURES = feature_info["features"]

TARGET = feature_info["target"]

CATEGORICAL_COLUMNS = feature_info[
    "categorical_columns"
]


print("\nFinal model loaded.")
print(
    "Model:",
    type(model).__name__
)

print(
    "Target:",
    TARGET
)

print(
    "Features:",
    FEATURES
)

print(
    "Categorical columns:",
    CATEGORICAL_COLUMNS
)


# ==============================================================
# LOAD FARMER INTELLIGENCE
# ==============================================================

final_intelligence_df = pd.read_csv(
    INTELLIGENCE_PATH
)


recommendations_df = pd.read_csv(
    RECOMMENDATION_PATH
)


district_intelligence_df = pd.read_csv(
    DISTRICT_INTELLIGENCE_PATH
)


forecast_df = pd.read_csv(
    FORECAST_PATH
)


# ==============================================================
# PREPARE FORECAST DATA
# ==============================================================

forecast_df["Date"] = pd.to_datetime(
    forecast_df["Date"],
    errors="coerce"
)

forecast_df["Predicted_Price"] = pd.to_numeric(
    forecast_df["Predicted_Price"],
    errors="coerce"
)

forecast_df = forecast_df.dropna(
    subset=[
        "Date",
        "Commodity",
        "District",
        "Predicted_Price"
    ]
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


print(
    "Loaded final_farmer_intelligence.csv:",
    final_intelligence_df.shape
)

print(
    "Loaded farmer_recommendations.csv:",
    recommendations_df.shape
)

print(
    "Loaded district_mandi_intelligence.csv:",
    district_intelligence_df.shape
)

print(
    "Loaded xgboost_30_day_price_forecast.csv:",
    forecast_df.shape
)


# ==============================================================
# DROPDOWN DATA
# ==============================================================

states = [
    "Maharashtra"
]


commodity_groups = [
    "Vegetables",
    "Cereals"
]


commodities = [
    "Onion",
    "Potato",
    "Tomato",
    "Rice",
    "Wheat"
]


districts = [
    "Ahmednagar",
    "Chhatrapati Sambhajinagar",
    "Jalgaon",
    "Kolhapur",
    "Nagpur",
    "Nashik",
    "Pune",
    "Sangli",
    "Satara",
    "Solapur"
]


festivals = [
    "No Festival",
    "Republic Day",
    "Holi",
    "Good Friday",
    "Dr. Ambedkar Jayanti",
    "Mahavir Jayanti",
    "Maharashtra Day",
    "Bakrid",
    "Independence Day",
    "Ganesh Chaturthi",
    "Gandhi Jayanti",
    "Dussehra",
    "Diwali",
    "Guru Nanak Jayanti",
    "Christmas",
    "Eid-ul-Fitr",
    "Gudi Padwa"
]


# ==============================================================
# HOME ROUTE
# ==============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        states=states,
        commodity_groups=commodity_groups,
        commodities=commodities,
        districts=districts,
        festivals=festivals
    )


# ==============================================================
# ENCODER HELPER
# ==============================================================

def encode_value(
    column,
    value
):

    if column not in encoders:

        raise ValueError(
            f"Encoder not found for column: {column}"
        )

    encoder = encoders[column]

    try:

        return encoder.transform(
            [value]
        )[0]

    except Exception:

        available = list(
            getattr(
                encoder,
                "classes_",
                []
            )
        )

        raise ValueError(
            f"Invalid {column}: {value}. "
            f"Available values: {available}"
        )


# ==============================================================
# AUTOMATIC PRICE HISTORY FEATURES
# ==============================================================

def get_price_history_features(
    commodity,
    district,
    prediction_date
):

    group = historical_df[
        (
            historical_df["Commodity"]
            == commodity
        )
        &
        (
            historical_df["District"]
            == district
        )
        &
        (
            historical_df["Date"]
            < prediction_date
        )
    ].copy()


    group = group.sort_values(
        "Date"
    )


    if len(group) < 30:

        raise ValueError(
            f"Not enough historical price data for "
            f"{commodity} in {district} before "
            f"{prediction_date.date()}."
        )


    prices = (
        group["Modal_Price"]
        .astype(float)
        .reset_index(drop=True)
    )


    lag1 = prices.iloc[-1]

    lag2 = prices.iloc[-2]

    lag3 = prices.iloc[-3]


    rolling7 = (
        prices
        .tail(7)
        .mean()
    )


    rolling30 = (
        prices
        .tail(30)
        .mean()
    )


    print("\n========== AUTOMATIC PRICE HISTORY ==========")

    print(
        "Commodity:",
        commodity
    )

    print(
        "District:",
        district
    )

    print(
        "Prediction date:",
        prediction_date.date()
    )

    print(
        "Lag 1:",
        round(float(lag1), 2)
    )

    print(
        "Lag 2:",
        round(float(lag2), 2)
    )

    print(
        "Lag 3:",
        round(float(lag3), 2)
    )

    print(
        "Rolling 7:",
        round(float(rolling7), 2)
    )

    print(
        "Rolling 30:",
        round(float(rolling30), 2)
    )

    print(
        "============================================"
    )


    return {

        "Lag_1":
            float(lag1),

        "Lag_2":
            float(lag2),

        "Lag_3":
            float(lag3),

        "Rolling_7":
            float(rolling7),

        "Rolling_30":
            float(rolling30)

    }


# ==============================================================
# PREDICTION ROUTE
# ==============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ======================================================
        # READ FORM INPUT
        # ======================================================

        state = request.form.get(
            "state"
        )

        commodity_group = request.form.get(
            "commodity_group"
        )

        commodity = request.form.get(
            "commodity"
        )

        district = request.form.get(
            "district"
        )

        festival = request.form.get(
            "festival"
        )


        arrival = float(
            request.form.get(
                "arrival"
            )
        )

        max_temp = float(
            request.form.get(
                "max_temp"
            )
        )

        min_temp = float(
            request.form.get(
                "min_temp"
            )
        )

        rainfall = float(
            request.form.get(
                "rainfall"
            )
        )

        windspeed = float(
            request.form.get(
                "windspeed"
            )
        )

        holiday = int(
            request.form.get(
                "holiday"
            )
        )

        day = int(
            request.form.get(
                "day"
            )
        )

        month = int(
            request.form.get(
                "month"
            )
        )

        year = int(
            request.form.get(
                "year"
            )
        )


        # ======================================================
        # INPUT VALIDATION
        # ======================================================

        if arrival < 0:

            raise ValueError(
                "Arrival cannot be negative."
            )


        if rainfall < 0:

            raise ValueError(
                "Rainfall cannot be negative."
            )


        if windspeed < 0:

            raise ValueError(
                "Wind speed cannot be negative."
            )


        if month < 1 or month > 12:

            raise ValueError(
                "Month must be between 1 and 12."
            )


        if day < 1 or day > 31:

            raise ValueError(
                "Day must be between 1 and 31."
            )


        # ======================================================
        # CREATE DATE
        # ======================================================

        try:

            input_date = datetime(
                year,
                month,
                day
            )

        except ValueError:

            raise ValueError(
                "Invalid calendar date."
            )


        # ======================================================
        # AUTOMATIC HISTORY FEATURES
        # ======================================================

        price_history = (
            get_price_history_features(
                commodity,
                district,
                input_date
            )
        )


        lag1 = price_history[
            "Lag_1"
        ]

        lag2 = price_history[
            "Lag_2"
        ]

        lag3 = price_history[
            "Lag_3"
        ]

        rolling7 = price_history[
            "Rolling_7"
        ]

        rolling30 = price_history[
            "Rolling_30"
        ]


        # ======================================================
        # CALENDAR FEATURES
        # ======================================================

        dayofweek = input_date.weekday()

        weekend = (
            1
            if dayofweek >= 5
            else 0
        )

        quarter = (
            (month - 1) // 3
        ) + 1


        # ======================================================
        # ENCODE CATEGORICAL VARIABLES
        # ======================================================

        state_encoded = encode_value(
            "State",
            state
        )


        commodity_group_encoded = encode_value(
            "Commodity_Group",
            commodity_group
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


        # ======================================================
        # CREATE INPUT DATA
        # ======================================================

        input_data = {

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


        # ======================================================
        # MODEL INPUT
        # ======================================================

        model_input = pd.DataFrame(
            [input_data]
        )


        missing_features = [
            feature
            for feature in FEATURES
            if feature not in model_input.columns
        ]


        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing_features
                )
            )


        model_input = model_input[
            FEATURES
        ]


        print(
            "\n========== FINAL MODEL INPUT =========="
        )

        print(
            model_input.to_string(
                index=False
            )
        )

        print(
            "\nFeature count:",
            len(model_input.columns)
        )

        print(
            "Feature order:",
            list(model_input.columns)
        )


        # ======================================================
        # MODEL PREDICTION
        # ======================================================

        prediction = model.predict(
            model_input
        )[0]


        prediction = max(
            0,
            float(prediction)
        )


        print(
            "\nPredicted Modal Price:",
            round(
                prediction,
                2
            )
        )


        # ======================================================
        # FARMER INTELLIGENCE
        # ======================================================

        intelligence = None


        if not final_intelligence_df.empty:

            matching_intelligence = (
                final_intelligence_df[
                    (
                        final_intelligence_df[
                            "Commodity"
                        ]
                        == commodity
                    )
                    &
                    (
                        final_intelligence_df[
                            "District"
                        ]
                        == district
                    )
                ]
            )


            if not matching_intelligence.empty:

                intelligence = (
                    matching_intelligence
                    .iloc[0]
                    .to_dict()
                )


        # ======================================================
        # FARMER RECOMMENDATION
        # ======================================================

        recommendation = None


        if not recommendations_df.empty:

            matching_recommendation = (
                recommendations_df[
                    (
                        recommendations_df[
                            "Commodity"
                        ]
                        == commodity
                    )
                    &
                    (
                        recommendations_df[
                            "District"
                        ]
                        == district
                    )
                ]
            )


            if not matching_recommendation.empty:

                recommendation = (
                    matching_recommendation
                    .iloc[0]
                    .to_dict()
                )


        # ======================================================
        # DISTRICT INTELLIGENCE
        # ======================================================

        district_intelligence = None


        if not district_intelligence_df.empty:

            matching_district = (
                district_intelligence_df[
                    district_intelligence_df[
                        "District"
                    ]
                    == district
                ]
            )


            if not matching_district.empty:

                district_intelligence = (
                    matching_district
                    .iloc[0]
                    .to_dict()
                )


        # ======================================================
        # DISTRICT-SPECIFIC 30-DAY FORECAST
        # ======================================================

        commodity_forecast = []


        if not forecast_df.empty:

            forecast_copy = forecast_df[
                (
                    forecast_df[
                        "Commodity"
                    ]
                    == commodity
                )
                &
                (
                    forecast_df[
                        "District"
                    ]
                    == district
                )
            ].copy()


            forecast_copy = (
                forecast_copy
                .sort_values("Date")
            )


            for _, row in forecast_copy.iterrows():

                commodity_forecast.append({

                    "Date":
                        row["Date"].strftime(
                            "%d-%m-%Y"
                        ),

                    "Predicted_Price":
                        float(
                            row[
                                "Predicted_Price"
                            ]
                        )

                })


        print(
            "\nForecast records for",
            commodity,
            "-",
            district,
            ":",
            len(commodity_forecast)
        )


        # ======================================================
        # RESULT PAGE
        # ======================================================

        return render_template(

            "result.html",

            prediction=round(
                prediction,
                2
            ),

            commodity=commodity,

            district=district,

            intelligence=intelligence,

            recommendation=recommendation,

            district_intelligence=
                district_intelligence,

            forecast=
                commodity_forecast

        )


    # ==========================================================
    # VALIDATION ERROR
    # ==========================================================

    except ValueError as e:

        print(
            "\nVALIDATION ERROR:",
            str(e)
        )

        return render_template(
            "error.html",
            error_type="Invalid Input",
            error_message=str(e)
        ), 400


    # ==========================================================
    # GENERAL ERROR
    # ==========================================================

    except Exception as e:

        print(
            "\nPREDICTION ERROR:",
            repr(e)
        )

        return render_template(
            "error.html",
            error_type="Prediction Error",
            error_message=str(e)
        ), 500


# ==============================================================
# RUN APPLICATION
# ==============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FLASK SERVER STARTING")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )