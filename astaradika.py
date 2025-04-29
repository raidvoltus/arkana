import os
import json
import glob
import joblib
import logging
import hashlib
import random
import pandas as pd
import numpy as np
import yfinance as yf

from typing import Dict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from model_utils import train_lightgbm, train_xgboost, train_lstm
from indicator_utils import calculate_indicators
from feature_utils import prepare_features_and_labels
from telegram_utils import send_telegram_message

# === Konstanta ===
MIN_PROB = 0.6
BACKUP_CSV_PATH = "backup_predictions.csv"
STOCK_LIST = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "UNVR.JK"]  # Contoh

def get_stock_data(ticker: str, period="3mo", interval="1h") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    except Exception as e:
        logging.error(f"Gagal mengambil data {ticker}: {e}")
        return pd.DataFrame()

def get_latest_close(ticker: str) -> float:
    try:
        df = yf.download(ticker, period="1d", interval="1h", progress=False, threads=False)
        if df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna().iloc[-1]
    except Exception as e:
        logging.warning(f"{ticker}: Gagal ambil harga terakhir - {e}")
        return None

def is_stock_eligible(price: float, avg_volume: float, atr: float, ticker: str) -> bool:
    if price < 500 or avg_volume < 100_000 or atr < 5:
        logging.info(f"{ticker}: Tidak lolos filter (Harga: {price}, Volume: {avg_volume}, ATR: {atr})")
        return False
    return True
    
def calculate_probability(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    try:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        return max(0.0, 1.0 - mae / (y_test.max() - y_test.min()))
    except Exception as e:
        logging.error(f"Error menghitung probabilitas: {e}")
        return 0.0

def log_prediction(ticker: str, tanggal: str, ph: float, pl: float, harga_awal: float):
    log_path = "prediksi_log.csv"
    with open(log_path, "a") as f:
        f.write(f"{ticker},{tanggal},{harga_awal:.2f},{ph:.2f},{pl:.2f}\n")

def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return None

    df = calculate_indicators(df)

    if "ATR" not in df.columns or df["ATR"].dropna().empty:
        logging.warning(f"{ticker}: ATR kosong setelah kalkulasi.")
        return None

    required_columns = ["High", "Low", "Close", "Volume", "ATR"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"{ticker}: Kolom yang diperlukan tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
        return None

    price = get_latest_close(ticker)
    if price is None:
        if df is not None and not df.empty and "Close" in df.columns:
            price = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else None
        else:
            logging.warning(f"{ticker}: Data fallback juga kosong.")
            return None

    if price is None:
        logging.warning(f"{ticker}: Tidak bisa mendapatkan harga terbaru.")
        return None
        
    df = calculate_indicators(df)
    df = df.dropna(subset=["ATR"])
    df = calculate_support_resistance(df)

    df = df.dropna(subset=[
        "ATR", "RSI", "MACD", "MACD_Hist", "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance", "VWAP", "ADX", "CCI",
        "Momentum", "WilliamsR", "OBV", "Stoch_K", "Stoch_D"
    ])

    # Tambahkan fitur waktu dan statistik harian
    df = add_time_features(df)
    df = add_statistical_features(df)

    # Drop baris terakhir untuk menghindari data leakage
    df_latest = df.iloc[-1:]
    df = df.iloc[:-1]

    features = [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist",
        "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance",
        "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
        "OBV", "Stoch_K", "Stoch_D", "Trend_Strength",
        "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]

    X_latest = df_latest[features]

    check_and_reset_model_if_needed(ticker, features)

    try:
        model_high_lgbm, _ = joblib.load(f"model_high_lgbm_{ticker}.pkl")
        model_low_lgbm, _ = joblib.load(f"model_low_lgbm_{ticker}.pkl")
    except Exception as e:
        logging.error(f"{ticker}: Gagal load model LightGBM - {e}")
        retrain_if_needed(ticker)
        try:
            model_high_lgbm, _ = joblib.load(f"model_high_lgbm_{ticker}.pkl")
            model_low_lgbm, _ = joblib.load(f"model_low_lgbm_{ticker}.pkl")
        except Exception as e2:
            logging.error(f"{ticker}: Gagal load model setelah retrain - {e2}")
            return None

    # Prediksi high dan low
    try:
        pred_high = float(model_high_lgbm.predict(X_latest)[0])
        pred_low = float(model_low_lgbm.predict(X_latest)[0])
    except Exception as e:
        logging.error(f"{ticker}: Error prediksi harga high/low - {e}")
        return None
        
def print_signal(signal: Dict[str, Any]):
    print("=" * 50)
    print(f"Sinyal Saham: {signal['ticker']}")
    print(f"Harga Saat Ini : {signal['harga']:.2f}")
    print(f"Take Profit    : {signal['take_profit']:.2f}")
    print(f"Stop Loss      : {signal['stop_loss']:.2f}")
    print(f"Potensi Profit : {signal['profit_potential_pct']:.2f}%")
    print(f"Probabilitas   : {signal['prob_success']*100:.1f}%")
    print(f"Aksi           : {signal['aksi']}")
    print("=" * 50)

def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram token atau chat_id tidak diset. Pesan tidak dikirim.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info("Pesan Telegram berhasil dikirim.")
        else:
            logging.warning(f"Gagal kirim pesan Telegram: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"Error saat kirim pesan Telegram: {e}")
        
def main():
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
        results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))

    if not results:
        logging.info("Tidak ada sinyal valid yang ditemukan.")
        return

    # Urutkan dan ambil sinyal terbaik
    sorted_results = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)
    top_signals = sorted_results[:5]

    for signal in top_signals:
        print_signal(signal)

    # Simpan hasil ke CSV
    pd.DataFrame(sorted_results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("Backup hasil analisis disimpan.")

    # Kirim pesan motivasi + sinyal via Telegram
    motivation = get_random_motivation()
    message = (
        f"<b>🔮Hai K.N.T.L. Clan Member🔮</b>\n"
        f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
        f"<b><i>{motivation}</i></b>\n\n"
        f"<b>Berikut Top 5 saham pilihan berdasarkan analisa K.N.T.L.A.I 🤖:</b>\n"
    )
    for s in top_signals:
        message += (
            f"\n🔹 {s['ticker']}\n"
            f"   💰 Harga: {s['harga']:.2f}\n"
            f"   🎯 TP: {s['take_profit']:.2f}\n"
            f"   🛑 SL: {s['stop_loss']:.2f}\n"
            f"   📈 Potensi Profit: {s['profit_potential_pct']:.2f}%\n"
            f"   ✅ Probabilitas: {s['prob_success']*100:.1f}%\n"
            f"   📌 Aksi: <b>{s['aksi'].upper()}</b>\n"
        )
    send_telegram_message(message)

def retrain_if_needed(ticker: str):
    akurasi_map = evaluate_prediction_accuracy()
    akurasi = akurasi_map.get(ticker, 1.0)

    if akurasi >= 0.90:
        logging.info(f"Akurasi {ticker} masih tinggi ({akurasi:.2%}), tidak perlu retrain.")
        return

    logging.info(f"Retraining model untuk {ticker} karena akurasi rendah ({akurasi:.2%})")

    df = get_stock_data(ticker)
    if df is None or df.empty:
        logging.error(f"{ticker}: Gagal ambil data untuk retraining.")
        return

    df = calculate_indicators(df).dropna(subset=["future_high", "future_low"])
    features = get_default_feature_set()

    try:
        X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = prepare_features_and_labels(df, features)
    except Exception as e:
        logging.error(f"{ticker}: Error saat mempersiapkan data retrain - {e}")
        return

    # LightGBM
    model_high_lgbm = train_lightgbm(X_tr, yh_tr)
    model_low_lgbm = train_lightgbm(X_tr, yl_tr)
    joblib.dump((model_high_lgbm, features), f"model_high_lgbm_{ticker}.pkl")
    joblib.dump((model_low_lgbm, features), f"model_low_lgbm_{ticker}.pkl")

    # XGBoost
    model_high_xgb = train_xgboost(X_tr, yh_tr)
    model_low_xgb = train_xgboost(X_tr, yl_tr)
    joblib.dump((model_high_xgb, features), f"model_high_xgb_{ticker}.pkl")
    joblib.dump((model_low_xgb, features), f"model_low_xgb_{ticker}.pkl")

    # LSTM
    model_high_lstm = train_lstm(X_tr, yh_tr)
    model_low_lstm = train_lstm(X_tr, yl_tr)
    model_high_lstm.save(f"model_high_lstm_{ticker}.keras")
    model_low_lstm.save(f"model_low_lstm_{ticker}.keras")

    # Simpan fitur ke JSON
    with open(f"{ticker}_features.json", "w") as f:
        json.dump(features, f)

    with open(f"features_lstm_{ticker}.json", "w") as f:
        json.dump(features, f)

    logging.info(f"{ticker}: Semua model retrain dan disimpan.")

def get_default_feature_set() -> List[str]:
    return [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist",
        "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance",
        "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
        "OBV", "Stoch_K", "Stoch_D", "Trend_Strength",
        "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]
    
def get_realized_price_data() -> pd.DataFrame:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File log prediksi tidak ditemukan.")
        return pd.DataFrame()

    df_log = pd.read_csv(log_path)
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
    results = []

    for ticker in df_log["ticker"].unique():
        df_ticker = df_log[df_log["ticker"] == ticker]
        start_date = df_ticker["tanggal"].min()
        end_date = df_ticker["tanggal"].max() + pd.Timedelta(days=6)

        try:
            df_price = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                threads=False
            )
        except Exception as e:
            logging.error(f"{ticker}: Gagal download data - {e}")
            continue

        if df_price.empty or not {"High", "Low"}.issubset(df_price.columns):
            logging.warning(f"{ticker}: Data harga kosong atau kolom penting hilang.")
            continue

        df_price.index = pd.to_datetime(df_price.index)

        for _, row in df_ticker.iterrows():
            window_start = row["tanggal"] + pd.Timedelta(days=1)
            window_end = row["tanggal"] + pd.Timedelta(days=6)
            df_window = df_price.loc[(df_price.index >= window_start) & (df_price.index <= window_end)]

            if df_window.shape[0] < 3:
                continue

            results.append({
                "ticker": ticker,
                "tanggal": row["tanggal"],
                "actual_high": df_window["High"].max(),
                "actual_low": df_window["Low"].min()
            })

    return pd.DataFrame(results)

def evaluate_prediction_accuracy() -> Dict[str, float]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("Log prediksi tidak tersedia.")
        return {}

    try:
        df_log = pd.read_csv(log_path, names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
    except Exception as e:
        logging.error(f"Gagal membaca log prediksi: {e}")
        return {}

    df_realized = get_realized_price_data()
    if df_realized.empty:
        logging.warning("Data realisasi harga kosong.")
        return {}

    df_realized["tanggal"] = pd.to_datetime(df_realized["tanggal"])
    df_merged = df_log.merge(df_realized, on=["ticker", "tanggal"], how="inner")

    if df_merged.empty:
        logging.info("Tidak ada data prediksi yang cocok dengan realisasi.")
        return {}

    df_merged["benar"] = (
        (df_merged["actual_high"] >= df_merged["pred_high"]) &
        (df_merged["actual_low"] <= df_merged["pred_low"])
    )

    akurasi_per_ticker = df_merged.groupby("ticker")["benar"].mean().to_dict()
    logging.info(f"Akurasi dihitung untuk {len(akurasi_per_ticker)} ticker.")

    return akurasi_per_ticker
    
def check_and_reset_model_if_needed(ticker: str, features: List[str]):
    hash_path = "model_feature_hashes.json"
    current_hash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()

    saved_hashes = {}
    if os.path.exists(hash_path):
        try:
            with open(hash_path, "r") as f:
                content = f.read().strip()
                if content:
                    saved_hashes = json.loads(content)
        except json.JSONDecodeError:
            logging.warning("File hash korup. Akan di-reset.")
            saved_hashes = {}

    if saved_hashes.get(ticker) != current_hash:
        logging.info(f"Perubahan fitur terdeteksi untuk {ticker}, model akan di-reset.")

        model_files = [
            f"model_high_lgb_{ticker}.pkl",
            f"model_low_lgb_{ticker}.pkl",
            f"model_high_xgb_{ticker}.pkl",
            f"model_low_xgb_{ticker}.pkl",
            f"model_lstm_{ticker}.keras"
        ]

        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
                logging.info(f"Model dihapus: {file}")

        saved_hashes[ticker] = current_hash
        with open(hash_path, "w") as f:
            json.dump(saved_hashes, f, indent=2)
    else:
        logging.info(f"Tidak ada perubahan fitur untuk {ticker}. Model tetap digunakan.")

def reset_models():
    patterns = [
        "model_high_lgb_*.pkl",
        "model_low_lgb_*.pkl",
        "model_high_xgb_*.pkl",
        "model_low_xgb_*.pkl",
        "model_lstm_*.keras"
    ]

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
        logging.info("Tidak ada model yang dihapus.")
    else:
        logging.info(f"Total model yang dihapus: {total_deleted}")
        
# === Daftar Kutipan Motivasi ===
MOTIVATION_QUOTES = [
    "Kegagalan adalah bumbu yang memberi rasa pada kesuksesan.",
    "Satu-satunya batasan kita untuk hari esok adalah keraguan kita hari ini.",
    "Kesuksesan tidak datang kepada yang menunggu, tapi kepada yang bekerja keras dan tidak menyerah.",
    "Setiap langkah kecil menuju tujuan adalah kemenangan besar.",
    "Ketakutan adalah musuh terbesar dari potensi kita."
]

def get_random_motivation() -> str:
    return random.choice(MOTIVATION_QUOTES)

# === Eksekusi & Kirim Sinyal ===
if __name__ == "__main__":
    reset_models()
    logging.info("🚀 Memulai analisis saham...")

    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]
    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Backup CSV disimpan")

    top_5 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:5]

    if top_5:
        motivation = get_random_motivation()
        message = (
            f"<b>🔮Hai K.N.T.L. Clan Member🔮</b>\n"
            f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
            f"<b><i>{motivation}</i></b>\n\n"
            f"<b>Berikut Top 5 saham pilihan berdasarkan analisa K.N.T.L.A.I 🤖:</b>\n"
        )

        for r in top_5:
            message += (
                f"\n🔹 {r['ticker']}\n"
                f"   💰 Harga: {r['harga']:.2f}\n"
                f"   🎯 TP: {r['take_profit']:.2f}\n"
                f"   🛑 SL: {r['stop_loss']:.2f}\n"
                f"   📈 Potensi Profit: {r['profit_potential_pct']:.2f}%\n"
                f"   ✅ Probabilitas: {r['prob_success'] * 100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )

        send_telegram_message(message)

    logging.info("✅ Selesai.")