"""
Project AeroSense: Model Development & Evaluation
"""

import numpy as np
import pandas as pd
from data_preprocessing import process_and_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def run_model_pipeline():
    X_train, X_test, y_train, y_test = process_and_split()
    
    print("\n--- Model Training Phase ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    
    lr_preds = lr.predict(X_test)
    rf_preds = rf.predict(X_test)
    
    print("\n--- Model Performance Evaluation ---")
    models = {"Linear Regression (Baseline)": lr_preds, "Random Forest (Advanced)": rf_preds}
    
    for name, preds in models.items():
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        print(f"\n[{name}]")
        print(f"  Mean Absolute Error (MAE)   : {mae:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"  R-squared (Accuracy Metric) : {r2:.4f}")
        
    importances = rf.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    print("\n--- Innovation Insights: Key Drivers of AQI ---")
    print(importance_df.to_string(index=False))
    
    plt.figure(figsize=(8, 4))
    plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='dodgerblue')
    plt.title("BASF Operational Dashboard: Metric Importance Ranking")
    plt.xlabel("Relative Influence Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

if __name__ == "__main__":
    run_model_pipeline()

