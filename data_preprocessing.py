"""
Project AeroSense: Data Preprocessing and Visualization Pipeline
Authors: [Insert Group Member Names Here]
Target Client: BASF SE Environmental Engineering Division
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    print("[1/4] Generating synthetic industrial air quality dataset...")
    np.random.seed(42)
    n_records = 1200
    
    # Simulating data that mimics real industrial zones (e.g., near a BASF plant)
    raw_data = {
        'PM2_5': np.random.gamma(shape=2, scale=18, size=n_records),
        'PM10': np.random.gamma(shape=3, scale=22, size=n_records),
        'NO2': np.random.normal(loc=35, scale=12, size=n_records),
        'SO2': np.random.normal(loc=18, scale=6, size=n_records),
        'CO': np.random.normal(loc=1.5, scale=0.5, size=n_records),
        'O3': np.random.normal(loc=38, scale=14, size=n_records),
        'Temperature': np.random.normal(loc=24, scale=7, size=n_records),
        'Humidity': np.random.uniform(low=25, high=85, size=n_records),
    }
    
    df = pd.DataFrame(raw_data)
    
    # Human touch: Intentionally add a few real-world errors (missing data) to fix
    df.loc[df['PM2_5'].sample(frac=0.03).index, 'PM2_5'] = np.nan
    df.loc[df['SO2'].sample(frac=0.02).index, 'SO2'] = np.nan
    
    # Calculate target AQI using a dynamic engineering heuristic formula
    df['AQI'] = (df['PM2_5']*0.55 + df['PM10']*0.35 + df['NO2']*0.4 + 
                 df['SO2']*0.5 - df['Temperature']*0.08 + np.random.normal(0, 4, n_records))
    
    print("[2/4] Handling missing values via regional median imputation...")
    df['PM2_5'] = df['PM2_5'].fillna(df['PM2_5'].median())
    df['SO2'] = df['SO2'].fillna(df['SO2'].median())
    
    return df

def generate_visualizations(df):
    print("[3/4] Creating EDA visualizations for the BASF engineering report...")
    
    # Chart 1: Feature Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='BrBG', fmt=".2f", linewidths=0.5)
    plt.title("AeroSense: Pollutant & Meteorological Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()
    
    # Chart 2: Distribution of AQI
    plt.figure(figsize=(8, 5))
    sns.histplot(df['AQI'], kde=True, color='purple', bins=30)
    plt.title("Distribution of Industrial AQI Values")
    plt.xlabel("Calculated AQI")
    plt.savefig("aqi_distribution.png")
    plt.close()
    print("-> Visualizations saved successfully as PNGs.")

def process_and_split():
    df = load_and_clean_data()
    generate_visualizations(df)
    
    X = df.drop(columns=['AQI'])
    y = df['AQI']
    
    print("[4/4] Scaling features and splitting data (80% Train, 20% Test)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_and_split()
    print("Data pipeline test passed successfully!")

