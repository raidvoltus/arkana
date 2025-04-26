import time
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import ta
from telegram import Bot
import os
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env
load_dotenv()

# Ambil API key dan token dari variabel lingkungan
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TELEGRAM_API_TOKEN = os.getenv('TELEGRAM_API_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
AI_MODEL_FILE = 'ai_model.pkl'
PAIR_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAME = '1h'
LOOKBACK_PERIOD = 100
ATR_PERIOD = 14
RISK_PERCENTAGE = 0.02
LIMIT_ORDER_OFFSET = 0.01
TRAILING_STOP_PERCENTAGE = 0.02
MIN_ATR = 0.01
USE_AI_CONFIRMATION = True
USE_DYNAMIC_POSITION_SIZING = True
MAX_QUANTITY = 10
QUANTITY = 1

# Setup logging
logging.basicConfig(level=logging.INFO)

# Helper Functions
def send_telegram_message(message):
    """Mengirim pesan ke Telegram."""
    bot = Bot(token=TELEGRAM_API_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message)

def get_candles(pair, timeframe, limit=100):
    """Mendapatkan data candlestick dari API Binance atau sumber lainnya."""
    # Ganti dengan panggilan API yang sesuai
    return pd.DataFrame()

def calculate_atr(df, period):
    """Menghitung ATR dari dataframe."""
    df["H-L"] = df["high"] - df["low"]
    df["H-C"] = abs(df["high"] - df["close"].shift(1))
    df["L-C"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=period).mean()
    return df["ATR"]

def train_ai_model(df):
    """Melatih model AI dengan RandomForest atau XGBoost."""
    if df is None or df.empty:
        logging.error("⚠️ Data tidak cukup untuk melatih AI.")
        return None  # Jangan lanjutkan jika data tidak ada

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    if len(df) < 50:  # Pastikan ada cukup data
        logging.warning("⚠️ Data terlalu sedikit untuk melatih AI.")
        return None

    features = ["ATR", "EMA9", "EMA21", "RSI"]
    X = df[features]
    y = df["target"]

    # Oversampling dengan SMOTE
    try:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    except ValueError as e:
        logging.warning(f"SMOTE tidak bisa dijalankan: {e}. Melanjutkan tanpa oversampling.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    rf_acc = model_rf.score(X_test, y_test)

    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model_xgb.fit(X_train, y_train)
    xgb_acc = model_xgb.score(X_test, y_test)

    logging.info(f"Akurasi RandomForest: {rf_acc:.2f}, Akurasi XGBoost: {xgb_acc:.2f}")

    return model_rf if rf_acc > xgb_acc else model_xgb

def predict_trend(df, model):
    """Memprediksi tren menggunakan model AI yang sudah dilatih."""
    features = ["ATR", "EMA9", "EMA21", "RSI"]
    if not all(feature in df.columns for feature in features):
        logging.error("DataFrame tidak memiliki semua fitur yang dibutuhkan untuk prediksi AI.")
        return None
    try:
        prediction = model.predict(df[features].iloc[-1].values.reshape(1, -1))[0]
        return prediction  # Kembalikan 0 atau 1
    except Exception as e:
        logging.error(f"Gagal memprediksi tren: {e}")
        return None

def get_best_trading_pair(pairs=PAIR_LIST):
    """Memilih pair dengan volatilitas tertinggi."""
    best_pair = None
    highest_volatility = 0

    for pair in pairs:
        df = get_candles(pair, TIMEFRAME, limit=LOOKBACK_PERIOD)
        if df is not None:
            df["ATR"] = calculate_atr(df, ATR_PERIOD)
            avg_volatility = df["ATR"].mean()

            if avg_volatility > highest_volatility:
                highest_volatility = avg_volatility
                best_pair = pair
        else:
            logging.warning(f"Tidak dapat memperoleh data untuk pair: {pair}")

    if best_pair:
        logging.info(f"Memilih {best_pair} sebagai pasangan perdagangan dengan volatilitas tertinggi.")
    else:
        logging.warning("Tidak dapat menemukan pasangan perdagangan yang sesuai. Menggunakan pasangan default.")
        best_pair = "BTCUSDT"
    return best_pair

# Main Bot Logic
def run_bot():
    global model
    send_telegram_message("🚀 Bot trading dimulai!")

    # Load model AI
    if os.path.exists(AI_MODEL_FILE):
        try:
            model = load(AI_MODEL_FILE)
            logging.info("✅ Model AI dimuat dari file.")
        except Exception as e:
            logging.error(f"Gagal memuat model AI: {e}")
            model = None

    if model is None:  # Jika model tidak ada, latih ulang
        logging.warning("⚠️ Model AI tidak tersedia, melatih ulang...")
        send_telegram_message("⚠️ Model AI tidak tersedia, melatih ulang...")
        # Latih model dengan data awal
        df = get_candles(get_best_trading_pair(), TIMEFRAME, limit=500)
        if df is not None:
            df["ATR"] = calculate_atr(df, ATR_PERIOD)
            df["EMA9"] = ta.trend.ema_indicator(df["close"], window=9, fillna=False)
            df["EMA21"] = ta.trend.ema_indicator(df["close"], window=21, fillna=False)
            df["RSI"] = ta.momentum.rsi(df["close"], window=14, fillna=False)
            model = train_ai_model(df)

            if model:
                dump(model, AI_MODEL_FILE)
                logging.info("Model AI disimpan ke file.")
            else:
                logging.warning("Tidak dapat melatih model AI.")
                send_telegram_message("⚠️ Tidak dapat melatih AI, mematikan konfirmasi AI.")
        else:
            logging.warning("Tidak dapat memperoleh data untuk melatih model AI.")
            send_telegram_message("⚠️ Tidak dapat melatih AI, mematikan konfirmasi AI.")

    while True:
        try:
            # Dapatkan pasangan terbaik
            best_pair = get_best_trading_pair()
            df = get_candles(best_pair, TIMEFRAME, limit=LOOKBACK_PERIOD)

            # Tambahkan indikator teknikal
            df["ATR"] = calculate_atr(df, ATR_PERIOD)
            df["EMA9"] = ta.trend.ema_indicator(df["close"], window=9, fillna=False)
            df["EMA21"] = ta.trend.ema_indicator(df["close"], window=21, fillna=False)
            df["RSI"] = ta.momentum.rsi(df["close"], window=14, fillna=False)

            # Prediksi tren dengan model AI
            if USE_AI_CONFIRMATION and model:
                trend_prediction = predict_trend(df, model)
                if trend_prediction == 1:
                    logging.info(f"Sinyal beli: {best_pair}")
                    send_telegram_message(f"Sinyal beli untuk {best_pair} dengan probabilitas naik tinggi.")
                else:
                    logging.info(f"Sinyal jual: {best_pair}")
                    send_telegram_message(f"Sinyal jual untuk {best_pair} dengan probabilitas turun tinggi.")

            # Evaluasi profit harian dan pengiriman
            # (Implementasi logika untuk menghitung dan mengirimkan profit harian dapat dilakukan di sini)
            
            time.sleep(300)  # Interval 5 menit, bisa dihapus jika ingin jalankan tanpa delay
        except Exception as e:
            logging.exception("Error in main loop:")
            send_telegram_message(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_bot()