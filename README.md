\# 🌾 Farmer's Price Intelligence



An end-to-end \*\*Agricultural Commodity Price Analysis and Prediction System\*\* that uses historical mandi prices, weather information, seasonal/event features, and machine learning to help understand and forecast agricultural commodity prices.



\## 📌 Project Overview



Farmer's Price Intelligence is a Data Science and Machine Learning project designed to analyze agricultural commodity prices across Maharashtra markets and provide useful intelligence for farmers and other stakeholders.



The system works with historical commodity-price data and engineered features such as lag prices, rolling averages, weather variables, seasonal information, festivals, holidays, arrivals, commodity groups, and districts.



The project combines \*\*data preprocessing, exploratory analysis, feature engineering, machine learning, forecasting, validation, clustering, risk analysis, and a Flask-based web application\*\*.



\---



\## 🎯 Objectives



\* Analyze historical mandi prices of agricultural commodities.

\* Identify important factors influencing commodity prices.

\* Build machine-learning models for price prediction.

\* Compare multiple machine-learning models.

\* Validate model performance across commodities and districts.

\* Generate short-term price forecasts.

\* Identify price-risk levels.

\* Provide farmer-oriented recommendations and intelligence.

\* Present the results through a web-based application.



\---



\## 🌾 Commodities



The project currently works with:



\* Onion

\* Potato

\* Rice

\* Tomato

\* Wheat



\---



\## 📍 Geographic Coverage



The dataset contains Maharashtra agricultural-market information covering multiple districts, including:



\* Ahmednagar

\* Pune

\* Nashik

\* Solapur

\* Nagpur

\* Kolhapur

\* Sangli

\* Satara

\* Jalgaon

\* Chhatrapati Sambhajinagar



The historical dataset used in the final modeling pipeline covers:



\*\*January 2023 – December 2025\*\*



\---



\## 📊 Dataset Features



The project uses agricultural, market, weather, and time-related features.



Important features include:



\* Commodity

\* Commodity Group

\* District

\* State

\* Arrival

\* Minimum Price

\* Maximum Price

\* Modal Price

\* Rainfall

\* Minimum Temperature

\* Maximum Temperature

\* Wind Speed

\* Festival

\* Holiday

\* Weekend

\* Day

\* Day of Week

\* Month

\* Quarter

\* Year

\* Lag 1

\* Lag 2

\* Lag 3

\* Rolling 7

\* Rolling 30



\---



\## 🔄 Data Science Pipeline



```text

Raw Agricultural Data

&#x20;       ↓

Data Loading

&#x20;       ↓

Data Cleaning

&#x20;       ↓

Data Integration

&#x20;       ↓

Weather \& Event Integration

&#x20;       ↓

Feature Engineering

&#x20;       ↓

Lag \& Rolling Features

&#x20;       ↓

Model Training

&#x20;       ↓

Model Validation

&#x20;       ↓

Model Comparison

&#x20;       ↓

Final Model Selection

&#x20;       ↓

Price Forecasting

&#x20;       ↓

Farmer Intelligence

&#x20;       ↓

Flask Web Application

```



\---



\## 🧹 Data Preprocessing



The preprocessing stage includes:



\* Cleaning agricultural price data.

\* Handling missing and inconsistent values.

\* Standardizing commodity and district information.

\* Integrating multiple datasets.

\* Preparing historical time-series data.

\* Adding weather-related variables.

\* Adding festival and holiday information.

\* Preparing data for machine-learning models.



\---



\## ⚙️ Feature Engineering



Several time-series and contextual features were created to improve prediction performance.



\### Lag Features



Historical prices were converted into lag variables:



```text

Lag\_1

Lag\_2

Lag\_3

```



These features allow the model to use previous price information when predicting future prices.



\### Rolling Features



Rolling price information was also generated:



```text

Rolling\_7

Rolling\_30

```



These represent short-term and longer-term historical price behavior.



\---



\## 🤖 Machine Learning Models



The project evaluates multiple machine-learning approaches, including:



\### 1. Random Forest



A Random Forest regression model was evaluated as a baseline/final comparison model.



\### 2. Improved XGBoost



An improved XGBoost model was trained using engineered agricultural and time-series features.



\### 3. Tuned XGBoost



Hyperparameter tuning was performed to improve the XGBoost model's predictive performance.



The \*\*Tuned XGBoost model\*\* was selected as the final price-prediction model.



\---



\## 🏆 Final Model Performance



The final model comparison produced the following results:



| Model                   |        MAE |       RMSE |         R² |

| ----------------------- | ---------: | ---------: | ---------: |

| Corrected Random Forest |     291.37 |     585.76 |     0.8303 |

| Improved XGBoost        |     244.86 |     449.02 |     0.9003 |

| \*\*Tuned XGBoost\*\*       | \*\*244.61\*\* | \*\*445.16\*\* | \*\*0.9020\*\* |



\### ⭐ Final Model



\*\*Tuned XGBoost\*\*



\* \*\*MAE:\*\* 244.61

\* \*\*RMSE:\*\* 445.16

\* \*\*R²:\*\* 0.9020



The final model achieved approximately \*\*90.2% R²\*\* on the final validation/test evaluation.



\---



\## 📈 Feature Importance



The final model identified the following important features:



| Rank | Feature         | Importance |

| ---: | --------------- | ---------: |

|    1 | Rolling\_7       |     0.7222 |

|    2 | Lag\_1           |     0.1457 |

|    3 | Lag\_2           |     0.0241 |

|    4 | Lag\_3           |     0.0171 |

|    5 | Commodity\_Group |     0.0157 |

|    6 | Arrival         |     0.0117 |

|    7 | Commodity       |     0.0083 |

|    8 | Festival        |     0.0082 |

|    9 | Rolling\_30      |     0.0081 |

|   10 | Holiday         |     0.0078 |



The results indicate that recent historical price behavior, especially the \*\*7-day rolling price\*\*, is highly influential in the final model.



\---



\## 🌾 Commodity-Level Validation



The final model was additionally evaluated separately across commodities.



This helps determine whether the model performs consistently for different agricultural commodities rather than relying only on the overall score.



The project generates:



```text

commodity\_mae\_comparison.png

```



and:



```text

phase6\_final\_by\_commodity.csv

```



\---



\## 📍 District-Level Validation



The model was also evaluated across individual districts.



The generated district-level evaluation includes:



\* MAE

\* RMSE

\* R²

\* Number of validation rows



Output files:



```text

district\_mae\_comparison.png

phase6\_final\_by\_district.csv

```



\---



\## 🔮 Price Forecasting



The project also generates short-term agricultural price forecasts.



A \*\*30-day price forecast\*\* is generated for the supported commodities.



The forecasting output is stored as:



```text

xgboost\_30\_day\_price\_forecast.csv

```



A forecast visualization is also generated:



```text

prophet\_30\_day\_forecasts.png

```



\---



\## 🚨 Farmer Intelligence



The project includes a Farmer Intelligence layer that combines model outputs and historical information to generate agricultural-market insights.



The system produces:



\* Farmer intelligence scores

\* Risk levels

\* District/mandi intelligence

\* Commodity-level recommendations

\* Farmer recommendations



Important outputs include:



```text

final\_farmer\_intelligence.csv

farmer\_recommendations.csv

district\_mandi\_intelligence.csv

cluster\_summary.csv

```



\---



\## 🖥️ Web Application



A Flask-based web application is included in the project.



Main application:



```text

app/app.py

```



The application provides a user interface for interacting with the Farmer's Price Intelligence system.



\### Application Structure



```text

app/

├── app.py

├── static/

│   └── style.css

└── templates/

&#x20;   ├── index.html

&#x20;   ├── result.html

&#x20;   └── error.html

```



\---



\## 📁 Project Structure



```text

Farmers\_Price\_Intelligence/

│

├── app/

│   ├── app.py

│   ├── static/

│   │   └── style.css

│   └── templates/

│       ├── index.html

│       ├── result.html

│       └── error.html

│

├── data/

│   └── raw/

│

├── integration/

│   └── farmer\_intelligence.py

│

├── models/

│   ├── 1\_XGBoost\_Price\_Forecaster.ipynb

│   ├── Model\_2\_Prophet Seasonal\_Model.ipynb

│   ├── Model\_3\_Random\_Forest\_Classifier.ipynb

│   ├── Model\_4\_KMeans\_Clustering.ipynb

│   ├── Model\_5\_Outlier\_Detector.ipynb

│   ├── Model\_6\_Crisis\_Score\_Engine.ipynb

│   ├── final\_feature\_list.json

│   ├── final\_label\_encoders.pkl

│   └── final\_price\_prediction\_model.pkl

│

├── notebooks/

│   ├── 00\_agmarknet\_downloader.ipynb

│   ├── 01\_dataloading.ipynb

│   ├── 02\_datacleaning.ipynb

│   ├── 03\_feature\_engineering.ipynb

│   ├── 04\_model\_training.ipynb

│   ├── 05\_model\_testing.ipynb

│   ├── 06\_wheather\_data.ipynb

│   ├── 07\_events\_data.ipynb

│   └── EDA.ipynb

│

├── outputs/

│   ├── final\_model\_comparison\_r2.png

│   ├── final\_model\_comparison\_mae.png

│   ├── final\_model\_comparison\_rmse.png

│   ├── final\_feature\_importance.png

│   ├── commodity\_mae\_comparison.png

│   └── district\_mae\_comparison.png

│

├── reports/

│   ├── final/

│   └── model\_predictions.csv

│

├── src/

│   ├── clean\_data.py

│   ├── final\_model\_graphs.py

│   ├── generate\_current\_30\_day\_forecast.py

│   ├── phase2\_xgboost.py

│   ├── phase3\_validation.py

│   ├── phase4\_lag\_rolling\_fix.py

│   ├── phase5\_final\_model.py

│   ├── phase6\_final\_validation.py

│   └── train\_model.py

│

├── .gitignore

└── README.md

```



\---



\## 🛠️ Technologies Used



\### Programming



\* Python

\* SQL



\### Data Science



\* Pandas

\* NumPy

\* Matplotlib

\* Scikit-learn



\### Machine Learning



\* XGBoost

\* Random Forest

\* K-Means Clustering

\* Outlier Detection

\* Time-series forecasting



\### Web Development



\* Flask

\* HTML

\* CSS



\### Development Tools



\* Jupyter Notebook

\* VS Code

\* Git

\* GitHub



\---



\## ▶️ Running the Project



Clone the repository:



```bash

git clone https://github.com/sanamujawar4569-lgtm/Farmers\_Price\_Intelligence.git

```



Move into the project:



```bash

cd Farmers\_Price\_Intelligence

```



Install the required Python dependencies:



```bash

pip install -r requirements.txt

```



Run the Flask application:



```bash

python app/app.py

```



Then open the local URL displayed by Flask in your browser.



\---



\## 📊 Final Reports and Visualizations



The final project contains model evaluation visualizations including:



\* Model R² comparison

\* Model MAE comparison

\* Model RMSE comparison

\* Feature importance

\* Commodity-wise MAE comparison

\* District-wise MAE comparison

\* 30-day forecast visualization



These are available under:



```text

outputs/

reports/final/

```



\---



\## 🔐 Data and Security



Sensitive information such as:



\* API keys

\* Passwords

\* Authentication tokens

\* Private credentials

\* Local environment files



should not be committed to the repository.



The project uses `.gitignore` to prevent unnecessary or sensitive files from being uploaded.



\---



\## 🚀 Future Scope



Future improvements could include:



\* Live AGMARKNET data integration.

\* More Maharashtra markets.

\* Additional agricultural commodities.

\* Real-time weather integration.

\* Improved commodity-specific models.

\* Explainable AI for individual predictions.

\* Mobile-friendly farmer interface.

\* Multilingual support including Marathi, Hindi, and Urdu.

\* Cloud deployment.

\* Automated daily price updates.

\* Real-time alerts for sudden price changes.



\---



\## 👩‍💻 Project Type



\*\*Data Science + Machine Learning + Flask Web Application\*\*



This project demonstrates an end-to-end workflow from raw agricultural data to machine-learning prediction and farmer-oriented intelligence.



\---



\## 📌 Author



\*\*Sanamujawar4569-lgtm\*\*



GitHub:



https://github.com/sanamujawar4569-lgtm



\---



\## ⭐ Project Highlights



\* 3 years of historical agricultural-market data

\* 5 agricultural commodities

\* Multiple Maharashtra districts

\* Weather and event features

\* Lag and rolling time-series features

\* Multiple ML models

\* Tuned XGBoost final model

\* \*\*R² ≈ 0.902\*\*

\* Commodity-level validation

\* District-level validation

\* 30-day forecasting

\* Farmer risk/intelligence engine

\* Flask web application

\* Complete GitHub project



