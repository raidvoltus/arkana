# === Import Libraries ===
import os, json, glob, joblib, hashlib, logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import lightgbm as lgb
import requests
from ta import trend, momentum, volatility, volume
from threading import Lock
from logging.handlers import RotatingFileHandler

# === Konstanta Umum ===
MIN_PRICE = 1000
MIN_VOLUME = 1000000
MIN_VOLATILITY = 0.01
HASH_PATH = "feature_hash.json"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
model_save_lock = Lock()

# === Logging Setup ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# === Kirim Telegram ===
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Logging Prediksi ===
def log_prediction(ticker: str, tanggal: str, pred_high: float, pred_low: float, harga_awal: float):
    with open("prediksi_log.csv", "a") as f:
        f.write(f"{ticker},{tanggal},{harga_awal},{pred_high},{pred_low}\n")

# === Ambil Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period="730d", interval="1h")
        if df is not None and not df.empty and len(df) >= 200:
            df["ticker"] = ticker
            return df
        logging.warning(f"Data kosong/kurang untuk {ticker}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    HOURS_PER_DAY = 6
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    df["MACD"] = trend.MACD(df["Close"]).macd()
    df["MACD_Hist"] = trend.MACD(df["Close"]).macd_diff()
    df["BB_Upper"] = volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["BB_Lower"] = volatility.BollingerBands(df["Close"]).bollinger_lband()
    df["Support"] = df["Low"].rolling(window=48).min()
    df["Resistance"] = df["High"].rolling(window=48).max()
    df["RSI"] = momentum.RSIIndicator(df["Close"]).rsi()
    df["SMA_14"] = trend.SMAIndicator(df["Close"], window=14).sma_indicator()
    df["SMA_28"] = trend.SMAIndicator(df["Close"], window=28).sma_indicator()
    df["SMA_84"] = trend.SMAIndicator(df["Close"], window=84).sma_indicator()
    df["EMA_10"] = trend.EMAIndicator(df["Close"], window=10).ema_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["CCI"] = trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()
    df["Momentum"] = momentum.ROCIndicator(df["Close"]).roc()
    df["WilliamsR"] = momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()
    df["hour"] = df.index.hour
    df["is_opening_hour"] = (df["hour"] == 9).astype(int)
    df["is_closing_hour"] = (df["hour"] == 14).astype(int)
    df["daily_avg"] = df["Close"].rolling(HOURS_PER_DAY).mean()
    df["daily_std"] = df["Close"].rolling(HOURS_PER_DAY).std()
    df["daily_range"] = df["High"].rolling(HOURS_PER_DAY).max() - df["Low"].rolling(HOURS_PER_DAY).min()
    df["future_high"] = df["High"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).max()
    df["future_low"]  = df["Low"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).min()
    return df.dropna()

# === Train Model LightGBM ===
def train_lightgbm(X, y) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model

# === Train LSTM ===
def train_lstm(X, y) -> Sequential:
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
    model.fit(X_arr, y, epochs=55, batch_size=32, verbose=0)
    return model

# === Hitung Probabilitas ===
def calculate_probability(model, X, y_true) -> float:
    y_pred = model.predict(X)
    close_price = X["Close"]
    correct_dir = ((y_pred > close_price) & (y_true > close_price)) | \
                  ((y_pred < close_price) & (y_true < close_price))
    return correct_dir.sum() / len(correct_dir.dropna()) if len(correct_dir.dropna()) else 0.0

# === Hash Fitur ===
def get_feature_hash(features: list[str]) -> str:
    return hashlib.md5(",".join(sorted(features)).encode()).hexdigest()

def check_and_reset_model_if_needed(ticker: str, features: list[str]):
    current_hash = get_feature_hash(features)
    try:
        with open(HASH_PATH, "r") as f:
            saved_hashes = json.load(f)
    except FileNotFoundError:
        saved_hashes = {}
    if saved_hashes.get(ticker) != current_hash:
        for suffix in ["high", "low"]:
            path = f"model_{suffix}_{ticker}.pkl"
            if os.path.exists(path): os.remove(path)
        if os.path.exists(f"model_lstm_{ticker}.keras"):
            os.remove(f"model_lstm_{ticker}.keras")
        saved_hashes[ticker] = current_hash
        with open(HASH_PATH, "w") as f:
            json.dump(saved_hashes, f, indent=2)

# === Analisis Ticker ===
def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None: return
    df = calculate_indicators(df)

    price = df["Close"].iloc[-1]
    avg_volume = df["Volume"].tail(20).mean()
    atr = df["ATR"].iloc[-1]
    if price < MIN_PRICE or avg_volume < MIN_VOLUME or (atr / price) < MIN_VOLATILITY:
        return

    features = [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist", "SMA_14", "SMA_28", "SMA_84",
        "EMA_10", "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX",
        "CCI", "Momentum", "WilliamsR", "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]
    check_and_reset_model_if_needed(ticker, features)
    df = df.dropna(subset=features + ["future_high", "future_low"])
    X, y_high, y_low = df[features], df["future_high"], df["future_low"]
    X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = train_test_split(X, y_high, y_low, test_size=0.2, random_state=42)

    # LightGBM
    high_path, low_path = f"model_high_{ticker}.pkl", f"model_low_{ticker}.pkl"
    model_high = joblib.load(high_path) if os.path.exists(high_path) else train_lightgbm(X_tr, yh_tr)
    model_low  = joblib.load(low_path) if os.path.exists(low_path) else train_lightgbm(X_tr, yl_tr)
    if not os.path.exists(high_path): joblib.dump(model_high, high_path)
    if not os.path.exists(low_path): joblib.dump(model_low, low_path)

    # LSTM
    lstm_path = f"model_lstm_{ticker}.keras"
    model_lstm = load_model(lstm_path) if os.path.exists(lstm_path) else train_lstm(X_tr, yh_tr)
    if not os.path.exists(lstm_path): model_lstm.save(lstm_path)

    # Prediksi dan log
    harga_awal = df["Close"].iloc[-1]
    pred_high = model_high.predict([X.iloc[-1]])[0]
    pred_low = model_low.predict([X.iloc[-1]])[0]
    log_prediction(ticker, datetime.today().strftime("%Y-%m-%d"), pred_high, pred_low, harga_awal)

    # Sinyal Telegram
    message = f"<b>{ticker}</b>\nHarga: {harga_awal:,.0f}\nPred High: {pred_high:,.0f}\nPred Low: {pred_low:,.0f}"
    send_telegram_message(message)

# === Evaluasi dan Retraining ===
def get_realized_price_data() -> pd.DataFrame:
    if not os.path.exists("prediksi_log.csv"): return pd.DataFrame()
    df_log = pd.read_csv("prediksi_log.csv", names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
    result = []
    for _, row in df_log.iterrows():
        ticker, tanggal = row["ticker"], pd.to_datetime(row["tanggal"])
        df = yf.download(ticker, start=tanggal.strftime("%Y-%m-%d"), end=(tanggal + pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
        if df.empty: continue
        result.append({"ticker": ticker, "tanggal": row["tanggal"], "actual_high": df["High"].max(), "actual_low": df["Low"].min()})
    return pd.DataFrame(result)

def evaluate_prediction_accuracy() -> Dict[str, float]:
    if not os.path.exists("prediksi_log.csv"): return {}
    df_log = pd.read_csv("prediksi_log.csv", names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
    df_real = get_realized_price_data()
    if df_real.empty: return {}
    df_merged = df_log.merge(df_real, on=["ticker", "tanggal"], how="inner")
    df_merged["benar"] = (df_merged["actual_high"] >= df_merged["pred_high"]) & (df_merged["actual_low"] <= df_merged["pred_low"])
    return df_merged.groupby("ticker")["benar"].mean().to_dict()

def retrain_if_needed(ticker: str):
    akurasi_map = evaluate_prediction_accuracy()
    if akurasi_map.get(ticker, 1.0) < 0.95:
        logging.info(f"Akurasi rendah, retraining {ticker}...")
        analyze_stock(ticker)

# === Main Eksekusi ===
if __name__ == "__main__":
    ticker_list = ["BBCA.JK", "TLKM.JK", "BBRI.JK"]  # Ganti sesuai kebutuhan
    for ticker in ticker_list:
        analyze_stock(ticker)
        retrain_if_needed(ticker)
