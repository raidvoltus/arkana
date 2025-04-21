# === Sinyal Saham Otomatis Full Fitur ===
import os, json, joblib, logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import lightgbm as lgb
import requests
from ta import trend, momentum, volatility, volume
from logging.handlers import RotatingFileHandler

MIN_PRICE = 1000
MIN_VOLUME = 1000000
MIN_VOLATILITY = 0.01
HASH_PATH = "feature_hash.json"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

log_handler = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[log_handler])

def send_telegram_message(message: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
        )
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True)
        if df is not None and not df.empty and len(df) >= 200:
            df["ticker"] = ticker
            return df
    except Exception as e:
        logging.error(f"Data error {ticker}: {e}")
    return None

def calculate_indicators(df):
    HOURS = 6
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["MACD"] = trend.MACD(df["Close"]).macd()
    df["MACD_Hist"] = trend.MACD(df["Close"]).macd_diff()
    df["BB_Upper"] = volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["BB_Lower"] = volatility.BollingerBands(df["Close"]).bollinger_lband()
    df["RSI"] = momentum.RSIIndicator(df["Close"]).rsi()
    df["SMA_14"] = trend.SMAIndicator(df["Close"], 14).sma_indicator()
    df["SMA_28"] = trend.SMAIndicator(df["Close"], 28).sma_indicator()
    df["EMA_10"] = trend.EMAIndicator(df["Close"], 10).ema_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["Momentum"] = momentum.ROCIndicator(df["Close"]).roc()
    df["hour"] = df.index.hour
    df["is_opening_hour"] = (df["hour"] == 9).astype(int)
    df["is_closing_hour"] = (df["hour"] == 14).astype(int)
    df["future_high"] = df["High"].shift(-HOURS).rolling(HOURS).max()
    return df.dropna()

def train_lightgbm(X, y):
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(X, y)
    return model

def train_lstm(X, y):
    X_arr = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_arr, y, epochs=50, batch_size=32, verbose=0)
    return model

def calculate_probability(model, X, y_true):
    y_pred = model.predict(X)
    close = X["Close"]
    correct = ((y_pred > close) & (y_true > close)) | ((y_pred < close) & (y_true < close))
    return correct.sum() / len(correct.dropna()) if len(correct.dropna()) else 0.0

def get_feature_hash(features):
    import hashlib
    return hashlib.md5(",".join(sorted(features)).encode()).hexdigest()

def check_model_hash(ticker, features):
    h = get_feature_hash(features)
    try:
        with open(HASH_PATH, "r") as f:
            hashes = json.load(f)
    except:
        hashes = {}
    if hashes.get(ticker) != h:
        for m in ["high", "lstm"]:
            path = f"model_{m}_{ticker}.pkl" if m == "high" else f"model_lstm_{ticker}.keras"
            if os.path.exists(path): os.remove(path)
        hashes[ticker] = h
        with open(HASH_PATH, "w") as f:
            json.dump(hashes, f)

def analyze_and_predict(ticker):
    df = get_stock_data(ticker)
    if df is None: return None
    df = calculate_indicators(df)

    price = df["Close"].iloc[-1]
    if price < MIN_PRICE or df["Volume"].mean() < MIN_VOLUME or df["ATR"].iloc[-1]/price < MIN_VOLATILITY:
        return None

    features = ["Close", "ATR", "MACD", "MACD_Hist", "RSI", "SMA_14", "SMA_28", "EMA_10", "BB_Upper", "BB_Lower", "VWAP", "ADX", "Momentum", "is_opening_hour", "is_closing_hour"]
    df = df.dropna(subset=features + ["future_high"])
    check_model_hash(ticker, features)

    X, y = df[features], df["future_high"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = f"model_high_{ticker}.pkl"
    model = joblib.load(model_path) if os.path.exists(model_path) else train_lightgbm(X_train, y_train)
    if not os.path.exists(model_path): joblib.dump(model, model_path)

    prob = calculate_probability(model, X_test, y_test)
    pred = model.predict([X.iloc[-1]])[0]
    profit_pct = round(((pred - price) / price) * 100, 2)

    return {
        "ticker": ticker,
        "harga": price,
        "pred_high": pred,
        "profit_pct": profit_pct,
        "probabilitas": round(prob, 2)
    }

def send_top_predictions():
    tickers = ["BBRI.JK", "BBCA.JK", "ANTM.JK", "UNVR.JK", "ASII.JK", "TLKM.JK"]
    predictions = []
    for t in tickers:
        result = analyze_and_predict(t)
        if result and (result["profit_pct"] >= 5 or result["probabilitas"] >= 0.75):
            predictions.append(result)
    top5 = sorted(predictions, key=lambda x: max(x["profit_pct"], x["probabilitas"]), reverse=True)[:5]
    if not top5: return

    msg = "<b>Sinyal Saham Top 5:</b>
"
    for p in top5:
        msg += f"<b>{p['ticker']}</b> Harga: {p['harga']:,.0f}, Pred: {p['pred_high']:,.0f}, Potensi: {p['profit_pct']}%, Prob: {p['probabilitas']}
"
    send_telegram_message(msg)

if __name__ == "__main__":
    send_top_predictions()
