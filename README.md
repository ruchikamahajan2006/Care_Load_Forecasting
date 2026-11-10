# 📊 Predictive Forecasting of Care Load & Placement Demand

### Machine Learning Based Time-Series Forecasting and Interactive Analytics Dashboard

## 📌 Project Overview

The **Predictive Forecasting of Care Load & Placement Demand** project aims to analyse historical data and predict future care load and discharge demand in the Unaccompanied Alien Children (UAC) Program.

The project uses Machine Learning and Time-Series Forecasting techniques to identify historical patterns, estimate future requirements, and provide early warnings for potential capacity pressure.

An interactive web dashboard built using Streamlit allows users to explore datasets, compare forecasting models, visualise predictions, and monitor future care demand.

## 🎯 Project Objectives

- Forecast the future number of children in HHS care.
- Estimate future discharge and placement demand.
- Analyse historical intake, transfer, and discharge trends.
- Compare statistical and machine learning forecasting approaches.
- Visualise future predictions and approximate prediction intervals.
- Identify potential capacity breaches through early-warning indicators.
- Support data-driven planning and resource allocation.

## 🗂️ Dataset Description

The project uses the HHS Unaccompanied Alien Children Program dataset.

### Dataset Features

| Column | Description |
|---|---|
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake volume |
| Children in CBP custody | Active CBP care load |
| Children transferred out of CBP custody | Transfers into the HHS system |
| Children in HHS Care | Active HHS care load |
| Children discharged from HHS Care | Discharge volume |

**Primary prediction targets:**
- Children in HHS Care
- Children discharged from HHS Care

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning |
| Statsmodels | Statistical forecasting |
| Plotly | Interactive visualisations |
| Streamlit | Web application |
| Git & GitHub | Version control and project hosting |

## 🤖 Machine Learning Models

The project implements and compares:

1. Naive Persistence Model
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. ARIMA / SARIMA (planned extension)

### Feature Engineering

- Lag features (1-day
