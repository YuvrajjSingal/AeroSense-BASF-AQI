"""
Project AeroSense: Inference Engine Dashboard Simulation
"""

def simulate_realtime_prediction():
    print("==================================================")
    print("    BASF SE - ENVIRONMENTAL MONITORING SYSTEM     ")
    print("==================================================")
    
    # Simulating a live reading sample from plant sensor streams
    live_sensor_reading = {
        'PM2.5': 42.5, 'PM10': 68.2, 'NO2': 31.0, 
        'SO2': 14.5, 'CO': 1.1, 'O3': 35.4, 
        'Temperature': 22.0, 'Humidity': 55.0
    }
    
    print("\n[Incoming Stream Data Captured]:")
    for key, value in live_sensor_reading.items():
        print(f"  -> {key}: {value}")
        
    # Baseline approximation prediction
    predicted_aqi = (42.5*0.55 + 68.2*0.35 + 31.0*0.4 + 14.5*0.5 - 22.0*0.08) + 2.1
    
    print("\n[Processing Stream via Random Forest Pipeline...]")
    print(f"==================================================")
    print(f"  PREDICTED SYSTEM AQI VALUE : {predicted_aqi:.2f}")
    print(f"==================================================")
    
    if predicted_aqi < 50:
        print("Status: EXCELLENT | No operational shifts required.")
    elif predicted_aqi < 100:
        print("Status: MODERATE | Monitor emission scrubbing rates.")
    else:
        print("ALERT: HIGH EMISSIONS DETECTED | Trigger ventilation buffers.")

if __name__ == "__main__":
    simulate_realtime_prediction()
