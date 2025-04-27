import os
import glob
import pandas as pd
import logging
import random
import json
import joblib
import ta
import hashlib
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from typing import Dict

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Global Constants ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
BACKUP_CSV_PATH = "backup_results.csv"
STOCK_LIST = ["BBRI.JK", "BBCA.JK", "BMRI.JK", "TLKM.JK", "ASII.JK"]

# === Daftar Kutipan Motivasi ===
MOTIVATION_QUOTES = [
    "Seseorang yang pernah melakukan kesalahan dan tidak pernah memperbaikinya berarti ia telah melakukan satu kesalahan lagi. - Konfusius.",
    "Anda tidak akan pernah belajar sabar dan berani jika di dunia ini hanya ada kebahagiaan. - Helen Keller.",
    "Tidak apa-apa untuk merayakan kesuksesan, tapi lebih penting untuk memperhatikan pelajaran tentang kegagalan. – Bill Gates."
]

# === Fungsi Kirim Telegram ===
def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram token atau chat_id tidak tersedia.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Fungsi Reset Model ===
def reset_models():
    patterns = ["model_high_*.pkl", "model_low_*.pkl", "model_lstm_*.keras"]
    total_deleted = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                logging.info(f"Dihapus: {filepath}")
                total_deleted += 1
            except Exception as e:
                logging.error(f"Gagal menghapus {filepath}: {e}")
    if total_deleted == 0:
        logging.info("Tidak ada model yang ditemukan untuk dihapus.")
    else:
        logging.info(f"Total {total_deleted} model dihapus.")

# === Fungsi Motivasi Random ===
def get_random_motivation() -> str:
    return random.choice(MOTIVATION_QUOTES)

# === Fungsi Auto Tuning LightGBM dan XGBoost ===
def train_best_model(X, y, model_type="lgbm"):
    if model_type == "lgbm":
        model = LGBMClassifier()
        param_dist = {
            'num_leaves': [20, 31, 50],
            'max_depth': [-1, 5, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 500]
        }
    elif model_type == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        param_dist = {
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 500],
            'subsample': [0.5, 0.7, 1.0]
        }
    else:
        raise ValueError("model_type harus 'lgbm' atau 'xgb'")

    search = RandomizedSearchCV(model, param_distributions=param_dist,
                                n_iter=10, scoring='accuracy', cv=3, random_state=42)
    search.fit(X, y)
    return search.best_estimator_

# === Fungsi Train LSTM ===
def train_lstm(X, y):
    # Memastikan data memiliki dimensi yang benar
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Menambahkan dimensi ketiga
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1], 1)),  # Sesuaikan dengan dimensi input
        keras.layers.LSTM(64),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

# === Fungsi Ambil Data Saham ===
def fetch_stock_data(ticker):
    df = yf.download(ticker, period="730d", interval="1d", progress=False)
    if df.empty:
        return None
    df["return"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    
    # Indikator tambahan
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bollinger_h"] = bollinger.bollinger_hband()
    df["bollinger_l"] = bollinger.bollinger_lband()
    df["atr_14"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Commodity Channel Index (CCI)
    df["cci"] = ta.trend.CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()

    # Williams %R
    df["williams_r"] = ta.momentum.WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).williams_r()

    df.dropna(inplace=True)
    return df

# === Fungsi Analisa Saham Utama ===
def analyze_stock(ticker):
    try:
        df = fetch_stock_data(ticker)
        if df is None or len(df) < 30:
            return None

        feature_cols = ["return", "sma_5", "sma_20"]
        X_raw = df[feature_cols].values
        y = (df["Close"].shift(-1) > df["Close"]).astype(int).values[:-1]
        X_raw = X_raw[:-1]

        # Split untuk model 2D
        X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

        # Split untuk model 3D (LSTM)
        X_train_3d = X_train_2d.reshape((X_train_2d.shape[0], X_train_2d.shape[1], 1))
        X_test_3d = X_test_2d.reshape((X_test_2d.shape[0], X_test_2d.shape[1], 1))

        # Training
        lgbm_model = train_best_model(X_train_2d, y_train, model_type="lgbm")
        xgb_model = train_best_model(X_train_2d, y_train, model_type="xgb")
        lstm_model = train_lstm(X_train_3d, y_train)

        # Prediksi
        preds_lgbm = lgbm_model.predict(X_test_2d)
        preds_xgb = xgb_model.predict(X_test_2d)
        preds_lstm = (lstm_model.predict(X_test_3d) > 0.5).astype(int).flatten()

        final_preds = (preds_lgbm + preds_xgb + preds_lstm) >= 2
        acc = accuracy_score(y_test, final_preds)

        current_price = df["Close"].iloc[-1]
        take_profit = current_price * 1.05
        stop_loss = current_price * 0.95

        action = "BUY" if acc > 0.6 else "WAIT"

        return {
            "ticker": ticker,
            "harga": current_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "profit_potential_pct": 5.0,
            "prob_success": acc,
            "aksi": action
        }
    except Exception as e:
        logging.error(f"Gagal analisa {ticker}: {e}")
        return None

# === Fungsi Evaluasi Akurasi Prediksi ===
def evaluate_prediction_accuracy() -> Dict[str, float]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File prediksi_log.csv tidak ditemukan.")
        return {}

    try:
        df_log = pd.read_csv(log_path)
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
        accuracy_map = {}
        for ticker in df_log["ticker"].unique():
            df_ticker = df_log[df_log["ticker"] == ticker]
            correct_preds = (df_ticker["action"] == df_ticker["predicted"]).sum()
            total_preds = len(df_ticker)
            accuracy_map[ticker] = correct_preds / total_preds if total_preds > 0 else 1.0
        return accuracy_map
    except Exception as e:
        logging.error(f"Error saat evaluasi akurasi prediksi: {e}")
        return {}

# === Main Function ===
def main():
    logging.info("Memulai analisis saham...")

    with ThreadPoolExecutor() as executor:
        results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))

    if not results:
        logging.error("Tidak ada hasil analisis.")
        return

    results = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)
    top_n = 5
    top_signals = results[:top_n]

    # Backup ke CSV
    df_backup = pd.DataFrame(top_signals)
    df_backup.to_csv(BACKUP_CSV_PATH, index=False)
    logging.info(f"Hasil disimpan ke {BACKUP_CSV_PATH}")

    # Kirim Telegram
    top_5 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:5]
    if top_5:
        motivation = get_random_motivation()
        message = (
            f"<b>🔮Hai K.N.T.L. Clan Member🔮</b>\n"
            f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
            f"<b><i>{motivation}</i></b>\n\n"
            f"<b>Berikut Top 5 saham pilihan berdasarkan analisa K.N.T.L. AI:</b>\n"
        )
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
        send_telegram_message(message)

    logging.info("Analisis saham selesai.")

if __name__ == "__main__":
    main()
