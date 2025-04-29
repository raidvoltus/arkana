# === IMPORT & KONFIGURASI AWAL ===
import os
import glob
import json
import hashlib
import joblib
import logging
import random
import warnings
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import yfinance as yf

import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === KONFIGURASI GLOBAL ===
BACKUP_CSV_PATH = "backup_result.csv"
STOCK_LIST = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK",
    "BBNI.JK", "MDKA.JK", "ANTM.JK", "ADRO.JK", "CPIN.JK", "ICBP.JK",
    "INDF.JK", "KLBF.JK", "SMGR.JK", "PTBA.JK", "PGAS.JK", "TBIG.JK",
    "TKIM.JK", "TOWR.JK"
]

# === DAFTAR KUTIPAN MOTIVASI ===
MOTIVATION_QUOTES = [
    "Keberanian bukan ketiadaan rasa takut, tapi kemenangan atas rasa takut.",
    "Kesuksesan adalah hasil dari ketekunan harian yang sering diremehkan.",
    "Hidup terlalu singkat untuk menunggu momen yang sempurna.",
    "Jangan takut gagal, takutlah tidak mencoba.",
    "Setiap detik adalah kesempatan untuk berubah."
]

# === FUNGSI UTILITAS ===

def get_stock_data(ticker: str, period: str = "90d", interval: str = "1h") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"Gagal mengambil data untuk {ticker}: {e}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA_14"] = df["Close"].rolling(window=14).mean()
    df["SMA_28"] = df["Close"].rolling(window=28).mean()
    df["SMA_84"] = df["Close"].rolling(window=84).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()

    df["stddev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_14"] + (2 * df["stddev"])
    df["BB_Lower"] = df["SMA_14"] - (2 * df["stddev"])

    df["daily_range"] = df["High"] - df["Low"]
    df["daily_avg"] = df["Close"].rolling(window=5).mean()
    df["daily_std"] = df["Close"].rolling(window=5).std()

    df["Close_Lag1"] = df["Close"].shift(1)
    df["Close_Lag2"] = df["Close"].shift(2)
    df["Trend_Strength"] = df["Close"].pct_change().rolling(5).std()

    df["is_opening_hour"] = df.index.hour.isin([9, 10]).astype(int)
    df["is_closing_hour"] = df.index.hour.isin([14, 15]).astype(int)

    # Dummy indikator lainnya
    df["RSI"] = np.random.rand(len(df)) * 100
    df["MACD"] = np.random.randn(len(df))
    df["MACD_Hist"] = np.random.randn(len(df))
    df["Support"] = df["Low"].rolling(window=10).min()
    df["Resistance"] = df["High"].rolling(window=10).max()
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
    df["ADX"] = np.random.rand(len(df)) * 50
    df["CCI"] = np.random.randn(len(df)) * 100
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["WilliamsR"] = -100 * ((df["High"].rolling(14).max() - df["Close"]) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min()))
    df["OBV"] = np.sign(df["Close"].diff()) * df["Volume"]
    df["Stoch_K"] = np.random.rand(len(df)) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    return df

# === MODEL TRAINING ===

def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    train_data = lgb.Dataset(X, label=y)
    param = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }
    model = lgb.train(param, train_data, 100)
    return model

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    param = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = xgb.train(param, dtrain, 100)
    return model

def train_lstm(X: pd.DataFrame, y: pd.Series) -> tf.keras.Model:
    X = X.values.reshape(X.shape[0], X.shape[1], 1)  # Reshape untuk LSTM
    y = y.values

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)
    return model

# === ANALYSIS & EVALUATION ===

def analyze_stock(ticker: str) -> dict:
    # Ambil data saham
    df = get_stock_data(ticker)
    if df is None or df.empty:
        return {}

    # Kalkulasi indikator teknikal
    df = calculate_indicators(df)
    df = df.dropna(subset=["future_high", "future_low"])

    # Fitur dan target variabel
    features = [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist",
        "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance",
        "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
        "OBV", "Stoch_K", "Stoch_D",
        "Close_Lag1", "Close_Lag2",
        "Trend_Strength",
        "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]
    
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]

    # Prediksi menggunakan model terlatih (misalnya, LightGBM, XGBoost, LSTM)
    model_high_lgb = joblib.load(f"model_high_lgb_{ticker}.pkl")
    model_low_lgb = joblib.load(f"model_low_lgb_{ticker}.pkl")
    model_high_xgb = joblib.load(f"model_high_xgb_{ticker}.pkl")
    model_low_xgb = joblib.load(f"model_low_xgb_{ticker}.pkl")
    model_lstm = tf.keras.models.load_model(f"model_lstm_{ticker}.keras")

    pred_high_lgb = model_high_lgb.predict(X)
    pred_low_lgb = model_low_lgb.predict(X)
    pred_high_xgb = model_high_xgb.predict(xgb.DMatrix(X))
    pred_low_xgb = model_low_xgb.predict(xgb.DMatrix(X))
    pred_high_lstm = model_lstm.predict(X.values.reshape(X.shape[0], X.shape[1], 1))

    # Gabungkan prediksi
    pred_high = (pred_high_lgb + pred_high_xgb + pred_high_lstm.flatten()) / 3
    pred_low = (pred_low_lgb + pred_low_xgb + pred_high_lstm.flatten()) / 3

    # Hitung take profit, stop loss, dan profit potential
    take_profit = pred_high
    stop_loss = pred_low
    profit_potential_pct = ((take_profit - stop_loss) / stop_loss) * 100

    # Hitung probabilitas keberhasilan
    prob_success = 0.75  # Angka ini bisa disesuaikan berdasarkan model atau analisis lainnya

    return {
        "ticker": ticker,
        "harga": df["Close"].iloc[-1],
        "take_profit": take_profit[-1],
        "stop_loss": stop_loss[-1],
        "profit_potential_pct": profit_potential_pct[-1],
        "prob_success": prob_success,
        "aksi": "BUY" if prob_success > 0.7 else "HOLD"
    }

def retrain_if_needed(ticker: str):
    akurasi_map = evaluate_prediction_accuracy()
    akurasi = akurasi_map.get(ticker, 1.0)  # default 100%
    
    if akurasi < 0.90:
        logging.info(f"Akurasi model {ticker} rendah ({akurasi:.2%}), retraining...")
        
        # Ambil data saham
        df = get_stock_data(ticker)
        if df is None or df.empty:
            logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
            return
        
        # Kalkulasi indikator teknikal
        df = calculate_indicators(df)
        df = df.dropna(subset=["future_high", "future_low"])

        # Tentukan fitur yang akan digunakan
        features = [
            "Close", "ATR", "RSI", "MACD", "MACD_Hist",
            "SMA_14", "SMA_28", "SMA_84", "EMA_10",
            "BB_Upper", "BB_Lower", "Support", "Resistance",
            "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
            "OBV", "Stoch_K", "Stoch_D",
            "Close_Lag1", "Close_Lag2",
            "Trend_Strength",
            "daily_avg", "daily_std", "daily_range",
            "is_opening_hour", "is_closing_hour"
        ]
        
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        
        # Latih model LightGBM
        model_high_lgb = train_lightgbm(X, y_high)
        joblib.dump(model_high_lgb, f"model_high_lgb_{ticker}.pkl")
        
        model_low_lgb = train_lightgbm(X, y_low)
        joblib.dump(model_low_lgb, f"model_low_lgb_{ticker}.pkl")

        # Latih model XGBoost
        model_high_xgb = train_xgboost(X, y_high)
        joblib.dump(model_high_xgb, f"model_high_xgb_{ticker}.pkl")
        
        model_low_xgb = train_xgboost(X, y_low)
        joblib.dump(model_low_xgb, f"model_low_xgb_{ticker}.pkl")
        
        # Latih model LSTM
        model_lstm = train_lstm(X, y_high)  # Asumsi menggunakan y_high untuk LSTM
        model_lstm.save(f"model_lstm_{ticker}.keras")
        
        logging.info(f"Model untuk {ticker} telah dilatih ulang dan disimpan.")
    else:
        logging.info(f"Akurasi model {ticker} sudah cukup baik ({akurasi:.2%}), tidak perlu retraining.")

  # === BACKUP & SEND SIGNAL ===

def send_telegram_message(message: str):
    """ Mengirim pesan ke Telegram menggunakan bot API """
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logging.info("Pesan berhasil dikirim ke Telegram.")
        else:
            logging.error(f"Gagal mengirim pesan: {response.status_code}")
    except Exception as e:
        logging.error(f"Gagal mengirim pesan ke Telegram: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Reset model jika perlu
    reset_models()
    
    logging.info("🚀 Memulai analisis saham...")
    
    # Tentukan jumlah pekerja untuk paralelisme
    max_workers = min(8, os.cpu_count() or 1)
    
    # Proses analisis saham secara paralel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    # Filter hasil yang valid
    results = [r for r in results if r]

    # Simpan hasil analisis ke CSV sebagai backup
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Backup CSV disimpan")

    # Ambil 5 saham teratas berdasarkan potensi profit
    top_5 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:5]
    
    if top_5:
        # Ambil kutipan motivasi
        motivation = get_random_motivation()
        message = (
            f"<b>🔮Hai K.N.T.L. Clan Member🔮</b>\n"
            f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
            f"<b><i>{motivation}</i></b>\n\n"
            f"<b>Berikut Top 5 saham pilihan berdasarkan analisa K.N.T.L.A.I 🤖:</b>\n"
        )
        
        # Format pesan untuk tiap saham
        for r in top_5:
            message += (
                f"\n🔹 {r['ticker']}\n"
                f"   💰 Harga: {r['harga']:.2f}\n"
                f"   🎯 TP: {r['take_profit']:.2f}\n"
                f"   🛑 SL: {r['stop_loss']:.2f}\n"
                f"   📈 Potensi Profit: {r['profit_potential_pct']:.2f}%\n"
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        
        # Kirim pesan ke Telegram
        send_telegram_message(message)

    logging.info("✅ Selesai.")
