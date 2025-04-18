# === Standard Library Imports ===
import os
import glob
import time
import json
import random
import logging
import hashlib
import threading
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor

# === Third-Party Imports ===
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf

from ta import momentum, trend, volatility, volume
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# === Konfigurasi Bot ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7
BACKUP_CSV_PATH  = "stock_data_backup.csv"
HASH_PATH        = "features_hash.json"

# === Threshold Konstanta ===
MIN_PRICE     = 100
MIN_VOLUME    = 10000
MIN_VOLATILITY = 0.005
MIN_PROB      = 0.8

# === Daftar Saham ===
STOCK_LIST = [
    "ACES.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "AUTO.JK",
    "AVIA.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BBYB.JK", "BDKR.JK", "BFIN.JK", "BMRI.JK", "BMTR.JK",
    "BNGA.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BTPS.JK", "CMRY.JK", "CPIN.JK", "CTRA.JK", "DEWA.JK",
    "DSNG.JK", "ELSA.JK", "EMTK.JK", "ENRG.JK", "ERAA.JK", "ESSA.JK", "EXCL.JK", "FILM.JK", "GGRM.JK", "GJTL.JK",
    "GOTO.JK", "HEAL.JK", "HMSP.JK", "HRUM.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK", "INET.JK", "INKP.JK",
    "INTP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KIJA.JK", "KLBF.JK", "KPIG.JK", "LSIP.JK", "MAPA.JK",
    "MAPI.JK", "MARK.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "MIDI.JK", "MIKA.JK", "MNCN.JK", "MTEL.JK", "MYOR.JK",
    "NCKL.JK", "NISP.JK", "PANI.JK", "PGAS.JK", "PGEO.JK", "PNLF.JK", "PTBA.JK", "PTPP.JK", "PTRO.JK", "PWON.JK",
    "RAJA.JK", "SCMA.JK", "SIDO.JK", "SMGR.JK", "SMIL.JK", "SMRA.JK", "SRTG.JK", "SSIA.JK", "SSMS.JK", "SURI.JK",
    "TINS.JK", "TKIM.JK", "TLKM.JK", "TOBA.JK", "TOWR.JK", "TPIA.JK", "TRIN.JK", "TSPC.JK", "UNTR.JK", "UNVR.JK"
]

# === Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_file_handler = RotatingFileHandler("stock_analysis.log", maxBytes=5 * 1024 * 1024, backupCount=2)
log_file_handler.setFormatter(log_formatter)

logger = logging.getLogger("StockAnalysis")
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)

# === Fungsi Kirim Pesan ke Telegram ===
def send_telegram_message(message: str):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        response = requests.post(url, data=data)
        if not response.ok:
            logger.error(f"Gagal mengirim pesan Telegram: {response.text}")
    except Exception as e:
        logger.error(f"Exception saat mengirim pesan Telegram: {str(e)}")

# === Fungsi Validasi & Hash Path ===
def validate_paths():
    if not os.path.exists("models_lgbm"):
        os.makedirs("models_lgbm")
    if not os.path.exists("models_lstm"):
        os.makedirs("models_lstm")
    if not os.path.exists("scalers"):
        os.makedirs("scalers")
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

def compute_feature_hash(data: pd.DataFrame) -> str:
    features_str = data.drop(columns=["High", "Low", "Target_High", "Target_Low"], errors='ignore').to_string()
    return hashlib.sha256(features_str.encode()).hexdigest()

def save_feature_hash(symbol: str, feature_hash: str):
    hashes = {}
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, 'r') as file:
            hashes = json.load(file)
    hashes[symbol] = feature_hash
    with open(HASH_PATH, 'w') as file:
        json.dump(hashes, file)

def load_feature_hash(symbol: str) -> Optional[str]:
    if not os.path.exists(HASH_PATH):
        return None
    with open(HASH_PATH, 'r') as file:
        hashes = json.load(file)
    return hashes.get(symbol)

# === Fungsi Mengunduh dan Membersihkan Data ===
def download_stock_data(symbol: str, start: str = "2018-01-01", end: str = None) -> Optional[pd.DataFrame]:
    try:
        end = end or datetime.now().strftime("%Y-%m-%d")
        data = yf.download(symbol + ".JK", start=start, end=end)
        if data.empty:
            logger.warning(f"Tidak ada data untuk {symbol}")
            return None
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.dropna()
        return data
    except Exception as e:
        logger.error(f"Error saat mengunduh data {symbol}: {str(e)}")
        return None

# === Fungsi Tambahan Indikator Teknikal ===
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df = df.dropna()
    return df

# === Fungsi Pembuatan Target ===
def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['Target_High'] = df['High'].shift(-1)
    df['Target_Low'] = df['Low'].shift(-1)
    return df.dropna()

# === Fungsi untuk Persiapan Data Pelatihan ===
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Menggunakan kolom indikator teknikal dan harga untuk fitur
    features = df[['EMA5', 'EMA10', 'RSI', 'MACD', 'ATR', 'OBV']]
    target_high = df['Target_High']
    target_low = df['Target_Low']
    return features, target_high, target_low

# === Pelatihan Model LightGBM ===
def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.Booster:
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=100)
    return model

# === Pelatihan Model LSTM ===
def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

# === Simpan Model LightGBM ===
def save_lightgbm_model(model: lgb.Booster, filename: str) -> None:
    model.save_model(filename)

# === Simpan Model LSTM ===
def save_lstm_model(model: Sequential, filename: str) -> None:
    model.save(filename)

# === Muat Model LightGBM ===
def load_lightgbm_model(filename: str) -> lgb.Booster:
    model = lgb.Booster(model_file=filename)
    return model

# === Muat Model LSTM ===
def load_lstm_model(filename: str) -> Sequential:
    model = keras.models.load_model(filename)
    return model

# === Fungsi untuk Prediksi Harga Menggunakan Model LightGBM ===
def predict_with_lightgbm(model: lgb.Booster, X: pd.DataFrame) -> pd.Series:
    predictions = model.predict(X)
    return predictions

# === Fungsi untuk Prediksi Harga Menggunakan Model LSTM ===
def predict_with_lstm(model: Sequential, X: np.ndarray) -> np.ndarray:
    predictions = model.predict(X)
    return predictions

# === Fungsi untuk Mengirim Alert ke Telegram ===
def send_telegram_alert(message: str, telegram_token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")

# === Fungsi untuk Mengirim Alert jika Prediksi Harga Mencapai Target ===
def check_and_send_alert(predicted_high: float, predicted_low: float, current_price: float, telegram_token: str, chat_id: str) -> None:
    # Ambil perbandingan harga
    if predicted_high > current_price:
        message = f"ALERT: Predicted High Price for the next day is {predicted_high:.2f}, which is higher than the current price ({current_price:.2f})."
        send_telegram_alert(message, telegram_token, chat_id)
    elif predicted_low < current_price:
        message = f"ALERT: Predicted Low Price for the next day is {predicted_low:.2f}, which is lower than the current price ({current_price:.2f})."
        send_telegram_alert(message, telegram_token, chat_id)

# === Fungsi untuk Memeriksa Kinerja Model ===
def evaluate_model_performance(model, X: pd.DataFrame, y: pd.Series) -> float:
    # Hitung MSE atau metrik kinerja lainnya untuk mengevaluasi model
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return mse

# === Fungsi untuk Melakukan Retraining Model Jika Diperlukan ===
def retrain_model(model, X_train: pd.DataFrame, y_train: pd.Series, model_type: str) -> object:
    if model_type == 'lgb':
        # Retraining untuk model LightGBM
        model.fit(X_train, y_train)
    elif model_type == 'lstm':
        # Retraining untuk model LSTM (dengan beberapa penyesuaian pada arsitektur dan pengaturan)
        model.fit(X_train, y_train, epochs=10, batch_size=32)  # Sesuaikan jumlah epoch dan batch size sesuai kebutuhan
    else:
        raise ValueError("Unsupported model type for retraining")
    return model

# === Fungsi untuk Mengecek Performa Model dan Melakukan Retraining Jika Perlu ===
def check_and_retrain_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float, model_type: str, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    mse = evaluate_model_performance(model, X_test, y_test)
    print(f"Model performance MSE: {mse:.2f}")
    
    if mse > threshold:  # Jika MSE lebih tinggi dari ambang batas, lakukan retraining
        print("Model underperforming. Retraining the model...")
        model = retrain_model(model, X_train, y_train, model_type)
    return model

# === Fungsi untuk Menyimpan Model ke File ===
def save_model(model, model_name: str, model_type: str) -> None:
    # Tentukan direktori tempat model disimpan
    model_dir = f"models/{model_type}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Beri nama file model berdasarkan waktu saat ini
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    
    # Simpan model menggunakan joblib
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")

# === Fungsi untuk Memuat Model dari File ===
def load_model(model_name: str, model_type: str, timestamp: str) -> object:
    # Tentukan direktori tempat model disimpan
    model_dir = f"models/{model_type}"
    
    # Bangun path lengkap model yang akan dimuat berdasarkan timestamp
    model_filename = f"{model_name}_{timestamp}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    # Memuat model menggunakan joblib
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

# === Fungsi untuk Memilih Model Terbaik Berdasarkan Performa ===
def choose_best_model(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> object:
    best_model = None
    best_mse = float('inf')
    
    for model_name, model in models.items():
        mse = evaluate_model_performance(model, X_test, y_test)
        print(f"Performance of {model_name} MSE: {mse:.2f}")
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
    
    print(f"Best model selected with MSE: {best_mse:.2f}")
    return best_model

# === Fungsi untuk Mengirim Pemberitahuan ke Telegram ===
def send_telegram_message(token: str, chat_id: str, message: str) -> None:
    bot = telegram.Bot(token=token)
    bot.send_message(chat_id=chat_id, text=message)
    print(f"Message sent to Telegram chat {chat_id}: {message}")

# === Fungsi untuk Mengirim Pemberitahuan Performa Model ===
def send_model_performance_notification(token: str, chat_id: str, model_name: str, mse: float) -> None:
    message = f"Model {model_name} performed with MSE: {mse:.2f} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    send_telegram_message(token, chat_id, message)

# === Fungsi untuk Mengirim Pemberitahuan Hasil Prediksi ===
def send_prediction_notification(token: str, chat_id: str, predicted_high: float, predicted_low: float) -> None:
    message = f"Predicted High: {predicted_high:.2f}\nPredicted Low: {predicted_low:.2f}\nPrediction time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    send_telegram_message(token, chat_id, message)

# === Fungsi untuk Melatih Model LightGBM ===
def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
    # Membuat dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Parameter model LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }

    # Melatih model
    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, early_stopping_rounds=50)
    return model

# === Fungsi untuk Melatih Model LSTM ===
def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 10) -> tf.keras.Model:
    # Membuat model LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))

    # Mengkompilasi model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Melatih model
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# === Fungsi untuk Menghitung Performa Model ===
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    # Untuk LightGBM
    if isinstance(model, lgb.Booster):
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # Untuk LSTM
    elif isinstance(model, tf.keras.Model):
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return mse

import datetime

# === Fungsi untuk Membuat Prediksi ===
def make_predictions(model, X_test: pd.DataFrame, model_type: str = 'lightgbm') -> np.ndarray:
    """
    Fungsi untuk membuat prediksi harga saham menggunakan model LightGBM atau LSTM.
    """
    if model_type == 'lightgbm':
        return model.predict(X_test, num_iteration=model.best_iteration)
    elif model_type == 'lstm':
        return model.predict(X_test)

# === Fungsi untuk Mengambil Keputusan Trading ===
def trading_decision(predictions: np.ndarray, current_price: float, threshold: float = 0.02) -> str:
    """
    Fungsi untuk mengambil keputusan trading berdasarkan prediksi harga saham.
    Jika prediksi harga lebih tinggi dari harga saat ini, membeli saham.
    Jika prediksi harga lebih rendah dari harga saat ini, menjual saham.
    """
    decision = "Hold"
    for prediction in predictions:
        if prediction > current_price * (1 + threshold):
            decision = "Buy"
        elif prediction < current_price * (1 - threshold):
            decision = "Sell"
    
    return decision

# === Fungsi untuk Mengirim Pemberitahuan (Telegram) ===
def send_telegram_alert(message: str, bot_token: str, chat_id: str):
    """
    Fungsi untuk mengirim pemberitahuan ke Telegram menggunakan bot Telegram.
    """
    import requests
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {
        'chat_id': chat_id,
        'text': message
    }
    response = requests.get(url, params=params)
    return response

# === Fungsi untuk Mengecek Waktu dan Mengambil Keputusan ===
def check_and_trade(model, X_test: pd.DataFrame, current_price: float, model_type: str, bot_token: str, chat_id: str):
    """
    Fungsi untuk memeriksa waktu dan melakukan keputusan trading secara otomatis.
    """
    predictions = make_predictions(model, X_test, model_type)
    decision = trading_decision(predictions, current_price)
    
    # Mengirim alert Telegram dengan keputusan trading
    message = f"Keputusan Trading: {decision} | Prediksi Harga: {predictions[-1]} | Harga Saat Ini: {current_price}"
    send_telegram_alert(message, bot_token, chat_id)
    
    return decision

# === Fungsi untuk Mengambil Data Realisasi dari YFinance ===
def get_realized_price_data(log_csv: str = "predictions/prediksi_log.csv",
                            lookahead_days: int = 1) -> pd.DataFrame:
    """
    Mengambil harga aktual untuk periode setelah prediksi,
    sehingga bisa dibandingkan dengan nilai prediksi.
    """
    if not os.path.exists(log_csv):
        return pd.DataFrame()

    df_log = pd.read_csv(log_csv,
                         names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])

    records = []
    for _, row in df_log.iterrows():
        symbol = row["ticker"].replace(".JK", "")
        start  = row["tanggal"].strftime("%Y-%m-%d")
        end    = (row["tanggal"] + pd.Timedelta(days=lookahead_days)).strftime("%Y-%m-%d")

        df_fut = yf.download(f"{symbol}.JK", start=start, end=end, progress=False)
        if df_fut.empty:
            continue

        actual_high = df_fut["High"].max()
        actual_low  = df_fut["Low"].min()

        records.append({
            "ticker":      row["ticker"],
            "tanggal":     row["tanggal"],
            "actual_high": actual_high,
            "actual_low":  actual_low
        })

    return pd.DataFrame(records)


# === Fungsi untuk Mengevaluasi Akurasi Prediksi ===
def evaluate_prediction_accuracy(log_csv: str = "predictions/prediksi_log.csv",
                                 lookahead_days: int = 1) -> Dict[str, float]:
    """
    Menghitung akurasi per ticker berdasarkan perbandingan:
      - pred_high <= actual_high
      - pred_low  >= actual_low
    """
    df_real = get_realized_price_data(log_csv, lookahead_days)
    if df_real.empty:
        logger.warning("Tidak ada data realisasi untuk dievaluasi.")
        return {}

    df_log = pd.read_csv(log_csv,
                         names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])

    # Merge log & realisasi
    df_merged = (
        df_log
        .merge(df_real, on=["ticker", "tanggal"], how="inner")
        .assign(
            benar=lambda df: (
                (df["pred_high"] <= df["actual_high"]) &
                (df["pred_low"]  >= df["actual_low"])
            )
        )
    )

    # Hitung akurasi per ticker
    accuracy = (
        df_merged
        .groupby("ticker")["benar"]
        .mean()
        .to_dict()
    )

    return accuracy

akurasi = evaluate_prediction_accuracy()
for sym, acc in akurasi.items():
    logger.info(f"Akurasi {sym}: {acc:.2%}")

def main():
    # 1. Siapkan folder dan logging
    validate_paths()
    logger.info("🚀 Memulai pipeline trading bot")

    results: List[Dict] = []

    # 2. Proses setiap simbol saham
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in STOCK_LIST}
        for fut in futures:
            symbol = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error(f"Error pada {symbol}: {e}")

    if not results:
        logger.info("❌ Tidak ada sinyal valid hari ini.")
    else:
        # 3. Backup hasil ke CSV
        pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
        logger.info(f"✅ Backup disimpan di {BACKUP_CSV_PATH}")

        # 4. Pilih Top 5 sinyal berdasarkan potensi profit
        top5 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:5]
        message = build_top5_message(top5)
        send_telegram_message(message)

    # 5. Evaluasi akurasi prediksi sebelumnya
    acc_map = evaluate_prediction_accuracy()
    for sym, acc in acc_map.items():
        logger.info(f"Akurasi {sym}: {acc:.2%}")

if __name__ == "__main__":
    main()
