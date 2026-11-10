
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import load_and_clean_data
from forecasting import train_models, forecast_future


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Care Load Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. WEBSITE DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f8fc;
}

h1, h2, h3 {
    color: #101a35;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce5f0;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.04);
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. LOAD DATASET
# ==================================================

@st.cache_data
def get_data():
    return load_and_clean_data()


try:
    df = get_data()

except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.info(
        "Check your CSV file and data folder."
    )
    st.stop()


# Target columns

CARE = "Children in HHS Care"

DISCHARGE = "Children discharged from HHS Care"

required_columns = [CARE, DISCHARGE]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Available columns:", list(df.columns))
    st.stop()

if len(df) < 30:
    st.error("Insufficient data for forecasting.")
    st.stop()


# ==================================================
# 4. TRAIN MACHINE LEARNING MODELS
# ==================================================

@st.cache_resource
def get_model_results(target):

    results, models, X_test, y_test = train_models(
        df[target]
    )

    return results, models, X_test, y_test


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📊 CARE FORECASTING")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Dataset Analysis",
        "Care Load Forecast",
        "Placement Demand",
        "Model Comparison",
        "Early Warning"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Forecast Settings")

days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=7,
    max_value=60,
    value=14,
    step=7
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Predictive Forecasting of Care Load "
    "& Placement Demand"
)


# ==================================================
# 6. OVERVIEW PAGE
# ==================================================

if page == "Overview":

    st.title(
        "Predictive Forecasting of Care Load "
        "& Placement Demand"
    )

    st.write(
        "An intelligent forecasting dashboard "
        "for analysing future UAC care requirements."
    )

    st.markdown("---")

    # KPI CARDS

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Observations",
        f"{len(df):,}"
    )

    col2.metric(
        "Current HHS Care Load",
        f"{df[CARE].iloc[-1]:,.0f}"
    )

    col3.metric(
        "Latest Discharges",
        f"{df[DISCHARGE].iloc[-1]:,.0f}"
    )

    st.markdown("---")

    # Historical care load

    st.subheader("Historical HHS Care Load")

    fig = px.line(
        df,
        y=CARE,
        title="Children in HHS Care",
        template="plotly_white"
    )

    fig.update_traces(
        line_color="#2463eb",
        line_width=2
    )

    fig.update_layout(height=450)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Historical placement demand

    st.subheader("Historical Placement Demand")

    fig2 = px.line(
        df,
        y=DISCHARGE,
        title="Children Discharged from HHS Care",
        template="plotly_white"
    )

    fig2.update_traces(
        line_color="#16a34a"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
        "Historical data is used to understand trends "
        "and forecast future care requirements."
    )


# ==================================================
# 7. DATASET ANALYSIS PAGE
# ==================================================

elif page == "Dataset Analysis":

    st.title("Dataset Analysis")

    st.write(
        "Explore the historical UAC dataset."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", f"{len(df):,}")

    col2.metric("Total Columns", len(df.columns))

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.reset_index(),
        use_container_width=True
    )

    # Statistical summary

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    # Correlation

    st.subheader("Correlation Heatmap")

    correlation = df.corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="Blues",
        title="Feature Correlation",
        aspect="auto"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Select a column

    st.subheader("Individual Feature Analysis")

    selected_column = st.selectbox(
        "Select a column",
        df.columns.tolist()
    )

    fig2 = px.line(
        df,
        y=selected_column,
        title=f"Trend of {selected_column}",
        template="plotly_white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ==================================================
# 8. COMMON FORECAST FUNCTION
# ==================================================

def show_forecast(target, heading):

    st.title(heading)

    st.write(
        "Select a machine learning model "
        "to predict future values."
    )

    results, models, X_test, y_test = (
        get_model_results(target)
    )

    # Model selection

    model_name = st.selectbox(
        "Select Forecasting Model",
        list(models.keys())
    )

    model = models[model_name]

    # Model predictions

    predictions = model.predict(X_test)

    metrics = results[model_name]

    st.markdown("---")

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{metrics['MAE']:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{metrics['RMSE']:,.2f}"
    )

    col3.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%"
    )

    st.markdown("---")

    # Future forecast

    st.subheader(
        f"Next {days} Days Forecast"
    )

    future = forecast_future(
        df[target],
        model,
        days
    )

    # Approximate residual-based prediction band

    residuals = (
        np.asarray(y_test) -
        np.asarray(predictions)
    )

    lower_error = np.quantile(
        residuals, 0.025
    )

    upper_error = np.quantile(
        residuals, 0.975
    )

    future["Lower"] = np.maximum(
        0,
        future["Forecast"] + lower_error
    )

    future["Upper"] = np.maximum(
        future["Lower"],
        future["Forecast"] + upper_error
    )

    # Forecast chart

    fig = go.Figure()

    historical = df[target].tail(120)

    fig.add_trace(
        go.Scatter(
            x=historical.index,
            y=historical.values,
            name="Historical Data",
            mode="lines",
            line=dict(color="#2463eb", width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future["Date"],
            y=future["Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future["Date"],
            y=future["Lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(36,99,235,0.15)",
            name="Approximate Prediction Band"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future["Date"],
            y=future["Forecast"],
            name="Future Forecast",
            mode="lines+markers",
            line=dict(color="#16a34a", width=3)
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis_title="Date",
        yaxis_title="Number of Children",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Forecast table

    st.subheader("Forecast Results")

    st.dataframe(
        future.round(2),
        use_container_width=True
    )

    # Download CSV

    csv = future.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="care_forecast.csv",
        mime="text/csv"
    )

    st.caption(
        "The shaded band is an approximate historical "
        "residual-based prediction range, not a "
        "calibrated confidence interval."
    )


# ==================================================
# 9. CARE LOAD FORECAST PAGE
# ==================================================

if page == "Care Load Forecast":

    show_forecast(
        CARE,
        "Future HHS Care Load Forecast"
    )


# ==================================================
# 10. PLACEMENT DEMAND PAGE
# ==================================================

elif page == "Placement Demand":

    show_forecast(
        DISCHARGE,
        "Future Discharge Demand Forecast"
    )


# ==================================================
# 11. MODEL COMPARISON PAGE
# ==================================================

elif page == "Model Comparison":

    st.title("Machine Learning Model Comparison")

    st.write(
        "Compare the performance of different "
        "forecasting models."
    )

    target = st.selectbox(
        "Select Prediction Target",
        [CARE, DISCHARGE]
    )

    results, models, X_test, y_test = (
        get_model_results(target)
    )

    comparison = pd.DataFrame(results).T

    st.markdown("---")

    st.subheader("Performance Metrics")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    # MAE graph

    st.subheader("MAE Comparison")

    fig1 = px.bar(
        comparison.reset_index(),
        x="index",
        y="MAE",
        color="MAE",
        title="Mean Absolute Error",
        template="plotly_white"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # RMSE graph

    st.subheader("RMSE Comparison")

    fig2 = px.bar(
        comparison.reset_index(),
        x="index",
        y="RMSE",
        color="RMSE",
        title="Root Mean Squared Error",
        template="plotly_white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info(
        "Lower MAE, RMSE and MAPE generally indicate "
        "better performance on the test dataset."
    )


# ==================================================
# 12. EARLY WARNING PAGE
# ==================================================

elif page == "Early Warning":

    st.title("Early Warning & Capacity Monitoring")

    st.write(
        "Identify potential future capacity pressure "
        "using predicted care load."
    )

    st.markdown("---")

    results, models, X_test, y_test = (
        get_model_results(CARE)
    )

    model = models["Random Forest"]

    future = forecast_future(
        df[CARE],
        model,
        days
    )

    # Capacity setting

    st.subheader("Capacity Configuration")

    threshold = st.number_input(
        "Enter Maximum Care Capacity",
        min_value=1,
        value=3000,
        step=100
    )

    max_forecast = future["Forecast"].max()

    breach_days = int(
        (future["Forecast"] > threshold).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Maximum Forecast",
        f"{max_forecast:,.0f}"
    )

    col2.metric(
        "Capacity Threshold",
        f"{threshold:,}"
    )

    col3.metric(
        "Potential Breach Days",
        breach_days
    )

    st.markdown("---")

    # Warning message

    if breach_days > 0:

        st.error(
            "⚠️ Potential capacity breach detected! "
            "Review resource planning."
        )

    else:

        st.success(
            "No capacity breach predicted within "
            "the selected forecast horizon."
        )

    # Capacity graph

    st.subheader("Capacity Monitoring Graph")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=future["Date"],
            y=future["Forecast"],
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#2463eb", width=3)
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="Capacity Threshold"
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_title="Date",
        yaxis_title="Care Load"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Details

    st.subheader("Future Capacity Details")

    st.dataframe(
        future,
        use_container_width=True
    )


# ==================================================
# 13. FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <b>Predictive Forecasting of Care Load
        & Placement Demand</b><br>
        Machine Learning and Data Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)
