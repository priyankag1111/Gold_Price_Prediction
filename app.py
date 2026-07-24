# Install Streamlit if not already installed. This is placed at the top to ensure it runs before any imports.
# Note: !pip install commands are generally not needed in Streamlit Cloud, but helpful for local testing.
import io
import time
# import importlib # Not needed for deployment if the module is part of the repo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Assuming gold_forecasting.py will be in the same directory on Streamlit Cloud
import gold_forecasting
from gold_forecasting import (
    build_features,
    determine_d,
    load_data,
    run_lstm_model,
    run_regression_model,
    run_sarimax_model,
)

# importlib.reload(gold_forecasting) # Not needed for deployment

st.set_page_config(page_title="Gold Price Prediction", page_icon="🪙", layout="wide")


# --------------------------------------------------------------------------
# Caching wrappers — keyed on file bytes / params so repeat runs are instant
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_data(file_bytes: bytes) -> pd.DataFrame:
    # Use BytesIO for in-memory file handling, as in Colab
    return load_data(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_regression(feat_json: str, test_size: float):
    feat = pd.read_json(io.StringIO(feat_json), orient="split", convert_dates=["Date"])
    _, dates, y_true, y_pred, metrics, _ = run_regression_model(feat, test_size=test_size)
    return dates, y_true, y_pred, metrics


@st.cache_data(show_spinner=False)
def cached_sarimax(df_json: str, test_size: float, seasonal: bool, m: int, max_p: int, max_q: int):
    df = pd.read_json(io.StringIO(df_json), orient="split", convert_dates=["Date"])
    _, dates, y_true, y_pred, metrics, order, sorder = run_sarimax_model(
        df, test_size=test_size, seasonal=seasonal, m=m, max_p=max_p, max_q=max_q
    )
    return dates, y_true, y_pred, metrics, order, sorder


@st.cache_data(show_spinner=False)
def cached_lstm(df_json: str, lookback: int, test_size: float, epochs: int, batch_size: int):
    df = pd.read_json(io.StringIO(df_json), orient="split", convert_dates=["Date"])
    _, dates, y_true, y_pred, metrics = run_lstm_model(
        df, lookback=lookback, test_size=test_size, epochs=epochs, batch_size=batch_size
    )
    return dates, y_true, y_pred, metrics


def line_chart(dates, y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, name="Actual", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, name="Predicted", line=dict(width=2, dash="dash")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Gold Price (USD)",
                       height=420, margin=dict(t=50, b=30))
    return fig


def metrics_bar_chart(all_metrics: dict):
    labels = ["RMSE", "MAE", "MAPE"]
    fig = go.Figure()
    for name, m in all_metrics.items():
        fig.add_trace(go.Bar(name=name, x=labels, y=[m[k] for k in labels]))
    fig.update_layout(barmode="group", title="Model Comparison", yaxis_title="Error",
                       height=420, margin=dict(t=50, b=30))
    return fig


# --------------------------------------------------------------------------
# Sidebar — data + model controls
# --------------------------------------------------------------------------
st.sidebar.title("🪙 Gold Price Prediction")
st.sidebar.markdown("Upload your CSV, or use the bundled sample dataset.")

uploaded_file = st.sidebar.file_uploader(
    "Gold price CSV (Date, Price, Open, High, Low, Vol., Change %)", type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("General")
test_size = st.sidebar.slider("Test set size (fraction)", 0.1, 0.4, 0.2, 0.05)

st.sidebar.subheader("Linear Regression")
n_lags = st.sidebar.slider("Number of lag features", 3, 30, 10)

st.sidebar.subheader("SARIMAX")
seasonal = st.sidebar.checkbox("Search seasonal component", value=False,
                                help="Daily gold prices usually show little seasonality; "
                                     "leave off unless you suspect weekly/monthly cycles.")
seasonal_m = st.sidebar.number_input("Seasonal period (m)", 2, 30, 5, disabled=not seasonal)
max_p = st.sidebar.slider("Max AR order (p)", 0, 5, 2)
max_q = st.sidebar.slider("Max MA order (q)", 0, 5, 2)

st.sidebar.subheader("LSTM")
lookback = st.sidebar.slider("Lookback window (days)", 10, 90, 30)
epochs = st.sidebar.slider("Epochs", 5, 150, 30)
batch_size = st.sidebar.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)

st.sidebar.markdown("---")
models_to_run = st.sidebar.multiselect(
    "Models to run", ["Linear Regression", "SARIMAX", "LSTM"],
    default=["Linear Regression", "SARIMAX", "LSTM"],
)
run_button = st.sidebar.button("🚀 Run models", type="primary", width='stretch')


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Gold Price Prediction")
st.caption("Linear Regression · SARIMAX · LSTM — trained and compared on your own data.")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    source_label = uploaded_file.name
else:
    st.info("No file uploaded — using the bundled sample dataset (2013-2023). "
            "Upload your own CSV in the sidebar to use your data.")
    with open("Gold Price (2013-2023).csv", "rb") as f:
        file_bytes = f.read()
    source_label = "Gold Price (2013-2023).csv (bundled)"

try:
    df = cached_load_data(file_bytes)
except Exception as e:
    st.error(f"Couldn't parse this CSV: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Source", source_label)
col2.metric("Rows", f"{len(df):,}")
col3.metric("Date range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

with st.expander("Preview data"):
    st.dataframe(df.tail(10), width='stretch')

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df["Date"], y=df["Price"], line=dict(width=1.5)))
fig_hist.update_layout(title="Full Price History", xaxis_title="Date",
                        yaxis_title="Gold Price (USD)", height=350, margin=dict(t=50, b=30))
st.plotly_chart(fig_hist, width='stretch')

if not run_button:
    st.markdown("Set your options in the sidebar, then click **🚀 Run models**.")
    st.stop()

if not models_to_run:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

df_json = df.to_json(orient="split", date_format="iso")
all_metrics = {}
tabs = st.tabs(models_to_run)

for tab, name in zip(tabs, models_to_run):
    with tab:
        t0 = time.time()
        if name == "Linear Regression":
            with st.spinner("Training Linear Regression..."):
                feat = build_features(df, n_lags=n_lags)
                feat_json = feat.to_json(orient="split", date_format="iso")
                dates, y_true, y_pred, metrics = cached_regression(feat_json, test_size)
            all_metrics[name] = metrics
            st.plotly_chart(line_chart(dates, y_true, y_pred,
                                        "Linear Regression: Actual vs Predicted"),
                             width='stretch')

        elif name == "SARIMAX":
            with st.spinner("Searching SARIMAX order and forecasting walk-forward "
                             "(this can take up to a minute on large datasets)..."):
                dates, y_true, y_pred, metrics, order, sorder = cached_sarimax(
                    df_json, test_size, seasonal, int(seasonal_m), max_p, max_q
                )
            st.caption(f"Selected order: SARIMAX{order} x {sorder}")
            all_metrics[name] = metrics
            st.plotly_chart(line_chart(dates, y_true, y_pred,
                                        f"SARIMAX{order}: Actual vs Predicted"),
                             width='stretch')

        elif name == "LSTM":
            with st.spinner(f"Training LSTM for up to {epochs} epochs..."):
                dates, y_true, y_pred, metrics = cached_lstm(
                    df_json, lookback, test_size, epochs, batch_size
                )
            all_metrics[name] = metrics
            st.plotly_chart(line_chart(dates, y_true, y_pred, "LSTM: Actual vs Predicted"),
                             width='stretch')

        elapsed = time.time() - t0
        m = all_metrics[name]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE", f"{m['RMSE']:.2f}")
c2.metric("MAE", f"{m['MAE']:.2f}")
c3.metric("MAPE", f"{m['MAPE']:.2f}%")
c4.metric("R²", f"{m['R2']:.3f}")
c5.metric("Time", f"{elapsed:.1f}s")

st.markdown("---")
st.subheader("Model Comparison")
st.plotly_chart(metrics_bar_chart(all_metrics), width='stretch')

results_df = pd.DataFrame([{"Model": k, **v} for k, v in all_metrics.items()])
st.dataframe(results_df.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}",
                                       "MAPE": "{:.2f}", "R2": "{:.3f}"}),
             width='stretch')
st.download_button("Download metrics as CSV", results_df.to_csv(index=False),
                    file_name="gold_price_model_metrics.csv", mime="text/csv")

best_model = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
st.success(f"Best model by RMSE on this run: **{best_model}**")
