# AeroSense – AI-Based Air Quality Prediction System for BASF

AeroSense is an AI-based Air Quality Index (AQI) prediction system developed for the **AI for Engineers** project.

The system uses environmental parameters and a trained **Gradient Boosting Regression** model to predict AQI.

## Team

- Yuvraj Singal – 1025020093
- Rishav Bansal – 1025020065
- Abdul Samad – 1025020066
- Manthan Batra – 1025020092
- Harsh Banga – 1025020048

## Project Objective

To develop a machine learning system that can estimate AQI from air-quality and environmental parameters for industrial and urban air-quality monitoring.

## Input Parameters

The model uses:

- PM2.5
- PM10
- NO₂
- SO₂
- CO
- O₃
- Temperature
- Humidity
- Wind Speed

## Dataset

The final dataset contains **10,000 records**:

- 2,239 real-world records
- 7,761 synthetic records

The dataset is available in:

`data/final_dataset.csv`

## Machine Learning Models

The project evaluated:

1. Linear Regression
2. Random Forest
3. Gradient Boosting

### Best Model

**Gradient Boosting Regressor**

Performance on the test set:

- MAE: 24.282
- RMSE: 45.907
- R² Score: 0.8597

The trained model is stored at:

`models/aerosense_aqi_model.pkl`

## AQI Categories

| AQI | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderately Polluted |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

## Project Structure

```text
AeroSense-BASF-AQI/
│
├── data/
│   └── final_dataset.csv
│
├── models/
│   └── aerosense_aqi_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── feature_importance.csv
│   └── AeroSense_Model_Summary.csv
│
├── app.py
├── data_preprocessing.py
├── model_training.py
├── requirements.txt
└── README.md
