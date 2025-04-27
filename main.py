# === Import Library ===
import os
import glob
import json
import logging
import random
import joblib
import yfinance as yf
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense

# === Konstanta Global ===
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Ganti sesuai kebutuhan
BACKUP_CSV_PATH = "backup_signals.csv"
MIN_PROB = 0.6

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MOTIVATION_QUOTES = [
    "Seseorang yang pernah melakukan kesalahan dan tidak pernah memperbaikinya berarti ia telah melakukan satu kesalahan lagi. - Konfusius.",
    "Anda tidak akan pernah belajar sabar dan berani jika di dunia ini hanya ada kebahagiaan. - Helen Keller.",
    "Tidak apa-apa untuk merayakan kesuksesan, tapi lebih penting untuk memperhatikan pelajaran tentang kegagalan. – Bill Gates."
]

# === Fungsi Auto-Tuning LightGBM ===
def train_lightgbm(X, y):
    param_grid = {
        "num_leaves": [20, 31, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 200, 300],
        "max_depth": [-1, 10, 20]
    }
    model = lgb.LGBMRegressor()
    random_search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring="neg_mean_squared_error", cv=3, random_state=42)
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    return best_model

# === Fungsi Auto-Tuning XGBoost ===
def train_xgboost(X, y):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.05, 0.1]
    }
    model = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0)
    random_search = RandomizedSearchCV(model, param_grid, n_iter=5, scoring="neg_mean_squared_error", cv=3, random_state=42)
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    return best_model

# === Fungsi LSTM Model ===
def train_lstm(X, y, model_type="keras"):
    X_3d = np.expand_dims(X.values, axis=1)
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_3d.shape[1], X_3d.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_3d, y.values, epochs=10, batch_size=16, verbose=0)
    return model

# === Fungsi Ensemble Model (LightGBM + XGBoost + LSTM) ===
def train_ensemble(X, y):
    model_lgb = train_lightgbm(X, y)
    model_xgb = train_xgboost(X, y)
    model_lstm = train_lstm(X, y)

    def ensemble_predict(X_input):
        pred_lgb = model_lgb.predict(X_input)
        pred_xgb = model_xgb.predict(X_input)
        X_3d = np.expand_dims(X_input.values, axis=1)
        pred_lstm = model_lstm.predict(X_3d).flatten()
        return (pred_lgb + pred_xgb + pred_lstm) / 3

    return model_lgb, model_xgb, model_lstm, ensemble_predict

# === Fungsi Ambil Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="6mo", interval="1h", progress=False, threads=False)
        if df.empty:
            logging.warning(f"{ticker}: Data kosong saat download.")
            return None
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"{ticker}: Error saat mengambil data - {e}")
        return None

# === Fungsi Ambil Harga Penutupan Terbaru ===
def get_latest_close(ticker: str):
    try:
        data = yf.download(ticker, period="1d", interval="1h", progress=False, threads=False)
        if not data.empty:
            return data["Close"].dropna().iloc[-1]
    except Exception as e:
        logging.error(f"{ticker}: Error mengambil harga terbaru - {e}")
    return None

# === Kalkulasi Indicator Teknis ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()
    df["SMA_14"] = df["Close"].rolling(14).mean()
    df["SMA_28"] = df["Close"].rolling(28).mean()
    df["SMA_84"] = df["Close"].rolling(84).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
    df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["CCI"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"]).cci()
    df["Momentum"] = ta.momentum.MomentumIndicator(df["Close"]).momentum()
    df["WilliamsR"] = ta.momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"]).williams_r()
    
    df["daily_avg"] = df["Close"].rolling(24).mean()
    df["daily_std"] = df["Close"].rolling(24).std()
    df["daily_range"] = df["High"].rolling(24).max() - df["Low"].rolling(24).min()
    
    df["is_opening_hour"] = df.index.hour.isin([9, 10, 11]).astype(int)
    df["is_closing_hour"] = df.index.hour.isin([14, 15]).astype(int)
    
    # Target future high dan low 6 hari ke depan
    df["future_high"] = df["High"].shift(-6*24).rolling(6*24).max()
    df["future_low"] = df["Low"].shift(-6*24).rolling(6*24).min()
    
    return df

# === Fungsi Cek Eligibility Saham ===
def is_stock_eligible(price, avg_volume, atr, ticker):
    if price < 1000:
        logging.info(f"{ticker}: Harga terlalu rendah ({price})")
        return False
    if avg_volume < 100_000:
        logging.info(f"{ticker}: Volume terlalu kecil ({avg_volume})")
        return False
    if atr/price < 0.005:
        logging.info(f"{ticker}: ATR terlalu kecil ({atr/price:.2%})")
        return False
    return True

# === Fungsi Persiapkan Fitur dan Label ===
def prepare_features_and_labels(df: pd.DataFrame, features: list):
    X = df[features].dropna()
    y_high = df.loc[X.index, "future_high"]
    y_low = df.loc[X.index, "future_low"]
    
    X_train = X.iloc[:-24]  # Sisain data terakhir buat testing
    X_test  = X.iloc[-24:]
    y_high_train = y_high.iloc[:-24]
    y_high_test  = y_high.iloc[-24:]
    y_low_train  = y_low.iloc[:-24]
    y_low_test   = y_low.iloc[-24:]

    return X_train, X_test, y_high_train, y_high_test, y_low_train, y_low_test

# === Konstanta ===
MIN_PROB = 0.5  # Minimal probabilitas prediksi yang diterima

# === LightGBM Training ===
def train_lightgbm(X, y):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model

# === XGBoost Training ===
def train_xgboost(X, y):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X, y)
    return model

# === LSTM Training ===
def train_lstm(X, y, model_type="keras"):
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    
    model.fit(X, y, epochs=30, batch_size=16, verbose=0)
    return model

# === Model Loader & Trainer (Auto Reload) ===
def load_or_train_model(filepath, train_fn, X, y, model_type="sklearn"):
    if os.path.exists(filepath):
        try:
            if model_type == "keras":
                return keras.models.load_model(filepath)
            else:
                return joblib.load(filepath)
        except Exception as e:
            logging.warning(f"Model corrupt, retraining... ({e})")
    
    model = train_fn(X, y)
    if model_type == "keras":
        model.save(filepath)
    else:
        joblib.dump(model, filepath)
    return model

# === Kalkulasi Probabilitas ===
def calculate_probability(model, X_test, y_test):
    try:
        preds = model.predict(X_test)
        score = np.mean(np.abs(preds - y_test) / (np.abs(y_test) + 1e-8))
        prob = 1 - score
        return prob
    except Exception as e:
        logging.error(f"Error saat kalkulasi probabilitas: {e}")
        return 0.0

# === Model Ensemble (LightGBM + LSTM + XGBoost) ===
def ensemble_predict(models, X_last):
    preds = []
    for model, mtype in models:
        if mtype == "keras":
            pred = model.predict(X_last.values.reshape((1, 1, X_last.shape[1])), verbose=0)[0][0]
        else:
            pred = model.predict(X_last)[0]
        preds.append(pred)
    return np.mean(preds)

# === Finalisasi Prediksi dengan Ensemble ===
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
        return None

    price = get_latest_close(ticker)
    if price is None:
        if df is not None and not df.empty and "Close" in df.columns:
            price = df["Close"].dropna().iloc[-1]
        else:
            logging.warning(f"{ticker}: Data fallback juga kosong.")
            return None

    if price is None:
        logging.warning(f"{ticker}: Tidak bisa mendapatkan harga terbaru.")
        return None

    avg_volume = df["Volume"].tail(20).mean()
    atr = df["ATR"].iloc[-1]

    if not is_stock_eligible(price, avg_volume, atr, ticker):
        logging.debug(f"{ticker}: Tidak memenuhi kriteria awal.")
        return None

    features = [
        "Close", "ATR", "RSI", "MACD", "MACD_Hist",
        "SMA_14", "SMA_28", "SMA_84", "EMA_10",
        "BB_Upper", "BB_Lower", "Support", "Resistance",
        "VWAP", "ADX", "CCI", "Momentum", "WilliamsR",
        "daily_avg", "daily_std", "daily_range",
        "is_opening_hour", "is_closing_hour"
    ]

    check_and_reset_model_if_needed(ticker, features)

    try:
        X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = prepare_features_and_labels(df, features)
    except Exception as e:
        logging.error(f"{ticker}: Error saat mempersiapkan data - {e}")
        return None

    model_high_lgb = load_or_train_model(f"model_high_lgb_{ticker}.pkl", train_lightgbm, X_tr, yh_tr)
    model_low_lgb  = load_or_train_model(f"model_low_lgb_{ticker}.pkl",  train_lightgbm, X_tr, yl_tr)

    model_high_xgb = load_or_train_model(f"model_high_xgb_{ticker}.pkl", train_xgboost, X_tr, yh_tr)
    model_low_xgb  = load_or_train_model(f"model_low_xgb_{ticker}.pkl",  train_xgboost, X_tr, yl_tr)

    model_lstm_high = load_or_train_model(f"model_lstm_high_{ticker}.keras", train_lstm, X_tr, yh_tr, model_type="keras")
    model_lstm_low  = load_or_train_model(f"model_lstm_low_{ticker}.keras",  train_lstm, X_tr, yl_tr, model_type="keras")

    try:
        prob_high = calculate_probability(model_high_lgb, X_te, yh_te)
        prob_low  = calculate_probability(model_low_lgb,  X_te, yl_te)
    except Exception as e:
        logging.error(f"{ticker}: Error saat kalkulasi probabilitas - {e}")
        return None

    if prob_high < MIN_PROB or prob_low < MIN_PROB:
        logging.info(f"{ticker} dilewati: Prob rendah (H={prob_high:.2f}, L={prob_low:.2f})")
        return None

    X_last = df[features].iloc[[-1]]

    # Gunakan ensemble prediction
    models_high = [(model_high_lgb, "sklearn"), (model_high_xgb, "sklearn"), (model_lstm_high, "keras")]
    models_low  = [(model_low_lgb,  "sklearn"), (model_low_xgb,  "sklearn"), (model_lstm_low,  "keras")]

    ph = ensemble_predict(models_high, X_last)
    pl = ensemble_predict(models_low, X_last)

    action = "beli" if (ph - price) / price > 0.02 else "jual"
    profit_potential_pct = (ph - price) / price * 100 if action == "beli" else (price - pl) / price * 100
    prob_succ = (prob_high + prob_low) / 2

    if (ph - price) / price > 0.02:
        aksi = "beli"
        take_profit = ph
        stop_loss = pl
    else:
        aksi = "jual"
        take_profit = pl
        stop_loss = ph

    if aksi == "beli":
        if take_profit <= price or stop_loss >= price:
            return None
    else:
        if take_profit >= price or stop_loss <= price:
            return None

    if profit_potential_pct < 10:
        logging.info(f"{ticker} dilewati: potensi profit rendah ({profit_potential_pct:.2f}%)")
        return None

    tanggal = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_prediction(ticker, tanggal, ph, pl, price)

    return {
        "ticker": ticker,
        "harga": round(price, 2),
        "take_profit": round(take_profit, 2),
        "stop_loss": round(stop_loss, 2),
        "aksi": action,
        "prob_high": round(prob_high, 2),
        "prob_low": round(prob_low, 2),
        "prob_success": round(prob_succ, 2),
        "profit_potential_pct": round(profit_potential_pct, 2),
    }

# === Main Program Eksekusi ===
if __name__ == "__main__":
    reset_models()
    logging.info("🚀 Memulai analisis saham K.N.T.L.A.I...")

    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]

    if results:
        pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
        logging.info("✅ Backup CSV berhasil disimpan.")

        top_5 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:5]

        motivation = get_random_motivation()
        message = (
            f"<b>🔮 Hai K.N.T.L. Clan Member 🔮</b>\n"
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
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )

        send_telegram_message(message)
        logging.info("✅ Pesan Telegram berhasil dikirim.")
    else:
        logging.warning("⚠️ Tidak ada saham yang lolos kriteria.")

    logging.info("🏁 Selesai.")
