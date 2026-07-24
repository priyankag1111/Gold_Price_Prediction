
import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------
# 1. Data loading — tailored to this dataset's exact format
# --------------------------------------------------------------------------
def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.replace('"', "").str.strip(),
        errors="coerce",
    )


def _clean_volume(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "").str.replace('"', "").str.strip()
    multiplier = np.where(s.str.endswith("K"), 1_000,
                  np.where(s.str.endswith("M"), 1_000_000,
                  np.where(s.str.endswith("B"), 1_000_000_000, 1)))
    numeric_part = pd.to_numeric(s.str.replace(r"[KMB]$", "", regex=True), errors="coerce")
    return numeric_part * multiplier


def _clean_percent(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%", "").str.replace('"', "").str.strip()
    return pd.to_numeric(s, errors="coerce") / 100.0


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = _clean_numeric(df[col])
    df["Vol."] = _clean_volume(df["Vol."])
    df["Change %"] = _clean_percent(df["Change %"])

    df = df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)
    df["Vol."] = df["Vol."].ffill().bfill()
    df["Change %"] = df["Change %"].ffill().bfill()
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering for the regression model (no lookahead leakage)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, n_lags: int = 10) -> pd.DataFrame:
    feat = df.copy()

    for lag in range(1, n_lags + 1):
        feat[f"lag_{lag}"] = feat["Price"].shift(lag)

    feat["roll_mean_7"] = feat["Price"].shift(1).rolling(7).mean()
    feat["roll_std_7"] = feat["Price"].shift(1).rolling(7).std()
    feat["roll_mean_30"] = feat["Price"].shift(1).rolling(30).mean()
    feat["roll_mean_90"] = feat["Price"].shift(1).rolling(90).mean()

    feat["prev_open"] = feat["Open"].shift(1)
    feat["prev_high"] = feat["High"].shift(1)
    feat["prev_low"] = feat["Low"].shift(1)
    feat["prev_range"] = feat["prev_high"] - feat["prev_low"]
    feat["prev_vol"] = feat["Vol."].shift(1)
    feat["prev_change_pct"] = feat["Change %"].shift(1)

    feat["day_of_week"] = feat["Date"].dt.dayofweek
    feat["month"] = feat["Date"].dt.month
    feat["day_of_year"] = feat["Date"].dt.dayofyear

    feat = feat.drop(columns=["Open", "High", "Low", "Vol.", "Change %"])
    feat = feat.dropna().reset_index(drop=True)
    return feat


# --------------------------------------------------------------------------
# 3. Time series regression model (Linear Regression on engineered features)
# --------------------------------------------------------------------------
def run_regression_model(feat: pd.DataFrame, test_size: float = 0.2):
    feature_cols = [c for c in feat.columns if c not in ("Date", "Price")]
    X = feat[feature_cols].values
    y = feat["Price"].values
    dates = feat["Date"].values

    split_idx = int(len(feat) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = compute_metrics(y_test, preds)
    return model, dates_test, y_test, preds, metrics, feature_cols


# --------------------------------------------------------------------------
# 4. SARIMAX model (auto order search + exogenous OHLC/Volume regressors)
# --------------------------------------------------------------------------
def determine_d(series: pd.Series, max_d: int = 2) -> int:
    """Pick the smallest differencing order that makes the series stationary
    per the Augmented Dickey-Fuller test (p < 0.05)."""
    s = series.copy()
    for d in range(max_d + 1):
        pval = adfuller(s.dropna())[1]
        if pval < 0.05:
            return d
        s = s.diff()
    return max_d


def auto_order_search(y_train, exog_train, d: int, seasonal: bool = False, m: int = 5,
                       max_p: int = 3, max_q: int = 3, max_P: int = 1, max_Q: int = 1):
    """Small grid search over (p,q) [and optionally seasonal (P,D,Q,m)],
    picking the combination with the lowest AIC on the training set."""
    best_aic = np.inf
    best_order = (1, d, 1)
    best_seasonal_order = (0, 0, 0, 0)

    p_range = range(0, max_p + 1)
    q_range = range(0, max_q + 1)
    seasonal_combos = [(0, 0, 0, 0)]
    if seasonal:
        seasonal_combos = [
            (P, D, Q, m)
            for P in range(0, max_P + 1)
            for D in range(0, 2)
            for Q in range(0, max_Q + 1)
        ]

    for p, q in itertools.product(p_range, q_range):
        if p == 0 and q == 0:
            continue
        for seasonal_order in seasonal_combos:
            try:
                model = SARIMAX(
                    y_train, exog=exog_train, order=(p, d, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

    return best_order, best_seasonal_order, best_aic


def run_sarimax_model(df: pd.DataFrame, test_size: float = 0.2, seasonal: bool = False,
                       m: int = 5, max_p: int = 3, max_q: int = 3):
    exog_df = pd.DataFrame({
        "Open": df["Open"].shift(1),
        "High": df["High"].shift(1),
        "Low": df["Low"].shift(1),
        "Vol": df["Vol."].shift(1),
    }).dropna().reset_index(drop=True)

    y = df["Price"].iloc[-len(exog_df):].reset_index(drop=True)
    dates = df["Date"].iloc[-len(exog_df):].reset_index(drop=True)

    split_idx = int(len(y) * (1 - test_size))
    y_train, y_test = y[:split_idx], y[split_idx:]
    exog_train = exog_df[:split_idx]
    exog_test = exog_df[split_idx:]
    dates_test = dates[split_idx:].reset_index(drop=True)

    d = determine_d(y_train)
    order, seasonal_order, aic = auto_order_search(
        y_train, exog_train, d=d, seasonal=seasonal, m=m, max_p=max_p, max_q=max_q
    )
    # print(f"    Selected SARIMAX order={order}, seasonal_order={seasonal_order} (AIC={aic:.1f})") # Removed for Streamlit context

    model = SARIMAX(
        y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    preds = []
    current_res = res
    for i in range(len(y_test)):
        next_exog = exog_test.iloc[[i]]
        fc = current_res.forecast(steps=1, exog=next_exog)
        preds.append(fc.iloc[0])
        next_y = y_test.iloc[[i]]
        current_res = current_res.append(next_y, exog=next_exog, refit=False)

    preds = np.array(preds)
    metrics = compute_metrics(y_test.values, preds)
    return res, dates_test.values, y_test.values, preds, metrics, order, seasonal_order


# --------------------------------------------------------------------------
# 5. Deep learning model (multivariate LSTM: Price, Open, High, Low, Vol.)
# --------------------------------------------------------------------------
def make_sequences(scaled_features: np.ndarray, target_col_idx: int, lookback: int):
    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i - lookback:i, :])
        y.append(scaled_features[i, target_col_idx])
    return np.array(X), np.array(y)


def run_lstm_model(df: pd.DataFrame, lookback: int = 30, test_size: float = 0.2,
                    epochs: int = 30, batch_size: int = 32):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(RANDOM_SEED)

    feature_df = pd.DataFrame({
        "Price": df["Price"].values,
        "Open": df["Open"].shift(1).values,
        "High": df["High"].shift(1).values,
        "Low": df["Low"].shift(1).values,
        "Vol": df["Vol."].shift(1).values,
    }).dropna().reset_index(drop=True)

    dates = df["Date"].values[-len(feature_df):]
    target_col_idx = feature_df.columns.get_loc("Price")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(feature_df.values)

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(feature_df[["Price"]].values)

    split_idx = int(len(scaled) * (1 - test_size))
    train_scaled = scaled[:split_idx]
    test_scaled = scaled[split_idx - lookback:]

    X_train, y_train = make_sequences(train_scaled, target_col_idx, lookback)
    X_test, y_test = make_sequences(test_scaled, target_col_idx, lookback)

    n_features = feature_df.shape[1]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    preds_scaled = model.predict(X_test, verbose=0)
    preds = price_scaler.inverse_transform(preds_scaled).flatten()
    y_test_actual = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    dates_test = dates[split_idx: split_idx + len(y_test_actual)]

    metrics = compute_metrics(y_test_actual, preds)
    return model, dates_test, y_test_actual, preds, metrics


# --------------------------------------------------------------------------
# 6. Metrics + plotting helpers
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def plot_predictions(dates, y_true, y_pred, title, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_comparison(all_metrics: dict, out_path):
    labels = ["RMSE", "MAE", "MAPE"]
    model_names = list(all_metrics.keys())
    x = np.arange(len(labels))
    width = 0.8 / len(model_names)

    plt.figure(figsize=(9, 5))
    for i, name in enumerate(model_names):
        vals = [all_metrics[name][k] for k in labels]
        plt.bar(x + i * width - (len(model_names) - 1) * width / 2, vals, width, label=name)
    plt.xticks(x, labels)
    plt.ylabel("Error")
    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_full_history(df: pd.DataFrame, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Price"], linewidth=1)
    plt.title("Gold Price History (2013-2023)")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
