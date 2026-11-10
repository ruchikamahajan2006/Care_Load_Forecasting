
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# Create lag-based features
def create_features(series):

    data = pd.DataFrame({"value": series})

    data["lag_1"] = data["value"].shift(1)
    data["lag_7"] = data["value"].shift(7)
    data["lag_14"] = data["value"].shift(14)

    data["rolling_mean_7"] = (
        data["value"].shift(1).rolling(7).mean()
    )

    data["rolling_mean_14"] = (
        data["value"].shift(1).rolling(14).mean()
    )

    data["rolling_std_7"] = (
        data["value"].shift(1).rolling(7).std()
    )

    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month

    return data.dropna()

# Calculate evaluation metrics
def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = mean_absolute_percentage_error(
        actual, predicted
    ) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2)
    }

# Train and compare models
def train_models(series):

    features = create_features(series)

    X = features.drop(columns=["value"])
    y = features["value"]

    # Chronological 80:20 split
    split = int(len(features) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),

        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )
    }

    results = {}
    trained_models = {}

    # Baseline: previous day's value
    naive_prediction = X_test["lag_1"].values

    results["Naive"] = evaluate_model(
        y_test, naive_prediction
    )

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results[name] = evaluate_model(
            y_test, predictions
        )

        trained_models[name] = model

    return (
        results,
        trained_models,
        X_test,
        y_test
    )

# Generate future forecasts recursively
def forecast_future(series, model, days):

    history = list(series.values)

    dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    predictions = []

    for date in dates:

        lag_1 = history[-1]
        lag_7 = history[-7]
        lag_14 = history[-14]

        rolling_mean_7 = np.mean(history[-7:])
        rolling_mean_14 = np.mean(history[-14:])
        rolling_std_7 = np.std(history[-7:], ddof=1)

        features = pd.DataFrame([{
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": rolling_mean_14,
            "rolling_std_7": rolling_std_7,
            "day_of_week": date.dayofweek,
            "month": date.month
        }])

        prediction = model.predict(features)[0]

        # Counts cannot be negative
        prediction = max(0, prediction)

        predictions.append(prediction)

        history.append(prediction)

    return pd.DataFrame({
        "Date": dates,
        "Forecast": predictions
    })