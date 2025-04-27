# === Import Libraries ===
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
from concurrent.futures import ThreadPoolExecutor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Global Constants ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
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
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")
        
# === Fungsi Reset Model ===
def reset_models():
    patterns = [
        "model_high_*.pkl",
        "model_low_*.pkl",
        "model_lstm_*.keras"
    ]

    total_deleted = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                print(f"Dihapus: {filepath}")
                total_deleted += 1
            except Exception as e:
                print(f"Gagal menghapus {filepath}: {e}")
    
    if total_deleted == 0:
        print("Tidak ada model yang ditemukan untuk dihapus.")
    else:
        print(f"Total {total_deleted} model dihapus.")

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
    model = keras.Sequential([
        keras.layers.Input(shape=(X.shape[1], 1)),
        keras.layers.LSTM(64),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X.reshape((X.shape[0], X.shape[1], 1)), y, epochs=10, batch_size=16, verbose=0)
    return model

# === Fungsi Ambil Data Saham ===
def fetch_stock_data(ticker):
    df = yf.download(ticker, period="90d", interval="1d", progress=False)
    if df.empty:
        return None
    df["return"] = df["Close"].pct_change()
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_20"] = df["Close"].rolling(window=20).mean()
    df.dropna(inplace=True)
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Return
    df["return_1d"] = df["Close"].pct_change()

    # Simple Moving Averages
    df["sma_5"] = df["Close"].rolling(window=5).mean()
    df["sma_20"] = df["Close"].rolling(window=20).mean()

    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bollinger_h"] = bollinger.bollinger_hband()
    df["bollinger_l"] = bollinger.bollinger_lband()

    # ATR
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()

    df = df.dropna()

    return df

def get_feature_hash(feature_list: list[str]) -> str:
    features_str = ",".join(sorted(feature_list))
    return hashlib.md5(features_str.encode()).hexdigest()

def get_latest_close(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1d")

        if df is None or df.empty:
            logging.warning(f"{ticker}: Data daily kosong saat ambil harga terbaru.")
            return None

        close_price = df["Close"].dropna()
        if close_price.empty:
            logging.warning(f"{ticker}: Kolom Close kosong di data daily.")
            return None

        return close_price.iloc[-1]

    except Exception as e:
        logging.error(f"{ticker}: Gagal ambil harga terbaru - {e}")
        return None


# === Fungsi Analisa Saham Utama ===
def analyze_stock(ticker):
    try:
        df = fetch_stock_data(ticker)
        if df is None or len(df) < 30:
            return None

        X = df[["return", "sma_5", "sma_20"]].values
        y = (df["Close"].shift(-1) > df["Close"]).astype(int).values[:-1]
        X = X[:-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train tiga model
        lgbm_model = train_best_model(X_train, y_train, model_type="lgbm")
        xgb_model = train_best_model(X_train, y_train, model_type="xgb")
        lstm_model = train_lstm(X_train, y_train)

        # Prediksi dan Ensemble
        preds_lgbm = lgbm_model.predict(X_test)
        preds_xgb = xgb_model.predict(X_test)
        preds_lstm = (lstm_model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1))) > 0.5).astype(int).flatten()

        final_preds = (preds_lgbm + preds_xgb + preds_lstm) >= 2

        acc = accuracy_score(y_test, final_preds)

        current_price = df["Close"].iloc[-1]
        take_profit = current_price * 1.05
        stop_loss = current_price * 0.95
        profit_potential_pct = 5.0

        action = "BUY" if acc > 0.6 else "WAIT"

        return {
            "ticker": ticker,
            "harga": current_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "profit_potential_pct": profit_potential_pct,
            "prob_success": acc,
            "aksi": action
        }
    except Exception as e:
        logging.error(f"Gagal analisa {ticker}: {e}")
        return None

def ensemble_predict(models: list, X_input):
    preds = []

    for model in models:
        if isinstance(model, tf.keras.Model):
            pred = model.predict(X_input, verbose=0).flatten()
        else:
            pred = model.predict(X_input)
        preds.append(pred)

    final_pred = sum(preds) / len(preds)
    return final_pred

def main():
    results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))
    
    results = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)

    top_n = 5  
    top_signals = results[:top_n]

def evaluate_prediction_accuracy() -> Dict[str, float]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File prediksi_log.csv tidak ditemukan.")
        return {}

    try:
        df_log = pd.read_csv(log_path, names=["tanggal", "ticker", "action", "target", "predicted"], header=0)
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])

        accuracy_map = {}

        for ticker in df_log["ticker"].unique():
            df_ticker = df_log[df_log["ticker"] == ticker]
            correct_preds = (df_ticker["action"] == df_ticker["predicted"]).sum()
            total_preds = len(df_ticker)

            if total_preds > 0:
                accuracy_map[ticker] = correct_preds / total_preds
            else:
                accuracy_map[ticker] = 1.0

        return accuracy_map
    except Exception as e:
        logging.error(f"Error saat evaluasi akurasi prediksi: {e}")
        return {}

# === Main Function ===
if __name__ == "__main__":
    # Menjalankan main function yang akan mengambil data dan menganalisa saham
    logging.info("Memulai analisis saham...")
    
    with ThreadPoolExecutor() as executor:
        # Analisis saham untuk setiap ticker dalam STOCK_LIST
        results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))
        
        # Mengurutkan hasil berdasarkan potensi profit tertinggi
        results = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)
        
        # Menampilkan Top N sinyal (misalnya 5 sinyal terbaik)
        top_n = 5  # atau bisa diubah menjadi 1 jika hanya ingin satu sinyal terbaik
        top_signals = results[:top_n]
        
        for signal in top_signals:
            logging.info(f"Signal: {signal}")

    logging.info("Analisis saham selesai.")
   
