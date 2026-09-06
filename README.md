# Project AeroSense: Predictive Air Quality Modeling for BASF Plant Safety

## 👥 Group Project Submission
* **Target Enterprise:** BASF SE (Environmental Engineering Division)
* **Group Size:** 4-5 Students 
* **Group Members:** 
  * Yuvraj Singal (1025020093)
  * Rishav Bansal (1025020065)
  * Abdul Samad   (1025020066)
  * Manthan Batra (1025020092)
  * Harsh Banga   (1025020048)
* **Course Assignment:** Machine Learning Supervised Regression Module

---

## 1. Problem Understanding & Core Objectives
Industrial plants require hyper-local environmental forecasts to manage emissions footprint dynamically. This project builds a **Supervised Regression Architecture** to predict the precise **Air Quality Index (AQI)** for BASF urban-industrial areas. By analyzing incoming levels of PM2.5, PM10, toxic gasses (NO₂, SO₂, CO, O₃), and weather parameters, our model enables facility managers to adjust chemical processing metrics proactively before threshold breaches occur.

---

## 2. Pre-processing & Workflow Pipeline
Our data pipeline enforces institutional data cleaning patterns:
1. **Data Ingestion:** Simulates historical sensor feeds matching BASF operational envelopes.
2. **Imputation:** Missing sensor variables are handled via localized statistical median interpolation.
3. **Feature Scaling:** `StandardScaler` transformations normalize numeric variance across varying units of concentration.

### Technical Workflow Diagram
```text
[Sensor Inputs] ──► [Median Imputation] ──► [Standard Scaling] ──► [Baseline / Ensemble Training] ──► [Evaluations]
```

---

## 3. Performance Results & Engineering Takeaways
We compared a standard baseline approach to an advanced ensemble tree methodology:

| Performance Parameter | Linear Regression (Baseline) | Random Forest Regressor (Advanced) |
| :--- | :--- | :--- |
| **MAE** (Lower is Better) | ~3.20 | **~1.15** |
| **RMSE** (Safety Critical) | ~4.12 | **~1.58** |
| **$R^2$ Variance Score** | ~0.9105 | **~0.9855** |

**Conclusion:** The Random Forest algorithm drastically handles the complex chemical interaction tracking required by BASF, minimizing higher order errors (RMSE) effectively.
