import os
import ta
import glob
import time
import threading
import joblib
import hashlib
import json
import requests
import random
import logging
import datetime
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf
import keras_tuner as kt

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List, Tuple
from ta import momentum, trend, volatility, volume
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from kerastuner.tuners import RandomSearch

# === Konfigurasi Bot ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7
BACKUP_CSV_PATH  = "stock_data_backup.csv"
HASH_PATH = "features_hash.json"

# === Load data ===
df = pd.read_csv("stock_data_backup.csv", index_col=0, parse_dates=True)

# === Daftar Saham ===
STOCK_LIST = [
    "AALI.JK", "ABBA.JK", "ABMM.JK", "ACES.JK", "ACST.JK", "ADHI.JK", "ADMF.JK", "ADMG.JK", "ADRO.JK", "AGII.JK",
    "AGRO.JK", "AKRA.JK", "AKSI.JK", "ALDO.JK", "ALKA.JK", "ALMI.JK", "AMAG.JK", "AMRT.JK", "ANDI.JK", "ANJT.JK",
    "ANTM.JK", "APIC.JK", "APLN.JK", "ARNA.JK", "ARTA.JK", "ASII.JK", "ASJT.JK", "ASRI.JK", "ASSA.JK", "ATIC.JK",
    "AUTO.JK", "BABP.JK", "BACA.JK", "BAEK.JK", "BALI.JK", "BAPA.JK", "BAPI.JK", "BATA.JK", "BBCA.JK", "BBHI.JK",
    "BBKP.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BINA.JK", "BIPP.JK", "BISI.JK", "BJBR.JK", "BJTM.JK", "BKDP.JK",
    "BKSL.JK", "BKSW.JK", "BLTA.JK", "BLTZ.JK", "BLUE.JK", "BMAS.JK", "BMRI.JK", "BMSR.JK", "BMTR.JK", "BNBA.JK",
    "BNGA.JK", "BNII.JK", "BNLI.JK", "BOBA.JK", "BOGA.JK", "BOLT.JK", "BOSS.JK", "BPFI.JK", "BPII.JK", "BPTR.JK",
    "BRAM.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BSSR.JK", "BTEK.JK", "BTEL.JK", "BTON.JK", "BTPN.JK",
    "BTPS.JK", "BUDI.JK", "BUVA.JK", "BVSN.JK", "BYAN.JK", "CAKK.JK", "CAMP.JK", "CANI.JK", "CARS.JK", "CASA.JK",
    "CASH.JK", "CBMF.JK", "CEKA.JK", "CENT.JK", "CFIN.JK", "CINT.JK", "CITA.JK", "CITY.JK", "CLAY.JK", "CLEO.JK",
    "CLPI.JK", "CMNP.JK", "CMRY.JK", "CMPP.JK", "CNKO.JK", "CNTX.JK", "COCO.JK", "COWL.JK", "CPIN.JK", "CPRO.JK",
    "CSAP.JK", "CSIS.JK", "CTRA.JK", "CTTH.JK", "DEAL.JK", "DEFI.JK", "DEPO.JK", "DGIK.JK", "DIGI.JK", "DILD.JK",
    "DIVA.JK", "DKFT.JK", "DLTA.JK", "DMAS.JK", "DNAR.JK", "DOID.JK", "DSSA.JK", "DUCK.JK", "DUTI.JK", "DVLA.JK",
    "DYAN.JK", "EAST.JK", "ECII.JK", "EDGE.JK", "EKAD.JK", "ELSA.JK", "EMDE.JK", "EMTK.JK", "ENRG.JK", "ENZO.JK",
    "EPAC.JK", "ERA.JK", "ERAA.JK", "ESSA.JK", "ESTA.JK", "FAST.JK", "FASW.JK", "FILM.JK", "FISH.JK", "FITT.JK",
    "FLMC.JK", "FMII.JK", "FOOD.JK", "FORU.JK", "FPNI.JK", "GAMA.JK", "GEMS.JK", "GGRM.JK", "GJTL.JK", "GLVA.JK",
    "GOOD.JK", "GPRA.JK", "GSMF.JK", "GZCO.JK", "HDTX.JK", "HERO.JK", "HEXA.JK", "HITS.JK", "HKMU.JK", "HMSP.JK",
    "HOKI.JK", "HRUM.JK", "ICBP.JK", "IDPR.JK", "IFII.JK", "INAF.JK", "INAI.JK", "INCF.JK", "INCI.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "INPP.JK", "INTA.JK", "INTD.JK", "INTP.JK", "IPCC.JK", "IPCM.JK", "IPOL.JK",
    "IPTV.JK", "IRRA.JK", "ISAT.JK", "ITMG.JK", "JAST.JK", "JAWA.JK", "JGLE.JK", "JKON.JK", "JPFA.JK", "JSMR.JK",
    "KAEF.JK", "KARW.JK", "KBLI.JK", "KBLM.JK", "KDSI.JK", "KIAS.JK", "KIJA.JK", "KINO.JK", "KLBF.JK", "KMTR.JK",
    "LEAD.JK", "LIFE.JK", "LINK.JK", "LPKR.JK", "LPPF.JK", "LUCK.JK", "MAIN.JK", "MAPB.JK", "MAPA.JK", "MASA.JK",
    "MCAS.JK", "MDKA.JK", "MEDC.JK", "MFIN.JK", "MIDI.JK", "MIRA.JK", "MITI.JK", "MKNT.JK", "MLPL.JK", "MLPT.JK",
    "MNCN.JK", "MPPA.JK", "MPRO.JK", "MTDL.JK", "MYOR.JK", "NATO.JK", "NELY.JK", "NFCX.JK", "NISP.JK", "NRCA.JK",
    "OKAS.JK", "OMRE.JK", "PANI.JK", "PBID.JK", "PCAR.JK", "PDES.JK", "PEHA.JK", "PGAS.JK", "PJAA.JK", "PMJS.JK",
    "PNBN.JK", "PNLF.JK", "POLA.JK", "POOL.JK", "PPGL.JK", "PPRO.JK", "PSSI.JK", "PTBA.JK", "PTIS.JK", "PWON.JK",
    "RAJA.JK", "RDTX.JK", "REAL.JK", "RICY.JK", "RIGS.JK", "ROTI.JK", "SAME.JK", "SAPX.JK", "SCCO.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SIMP.JK", "SIPD.JK", "SMBR.JK", "SMCB.JK", "SMDR.JK", "SMGR.JK", "SMKL.JK", "SMRA.JK",
    "SMSM.JK", "SOCI.JK", "SQMI.JK", "SRAJ.JK", "SRTG.JK", "STAA.JK", "STTP.JK", "TALF.JK", "TARA.JK", "TBIG.JK",
    "TCID.JK", "TIFA.JK", "TINS.JK", "TKIM.JK", "TLKM.JK", "TOTO.JK", "TPIA.JK", "TRIM.JK", "TURI.JK", "ULTJ.JK",
    "UNIC.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK", "WSKT.JK", "YPAS.JK", "ZINC.JK"
]

# === Logging Setup ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler   = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)
def log_prediction(ticker: str, tanggal: str, pred_high: float, pred_low: float, harga_awal: float):
    with open("prediksi_log.csv", "a") as f:
        f.write(f"{ticker},{tanggal},{harga_awal},{pred_high},{pred_low}\n")

# === Lock untuk Thread-Safe Model Saving ===
model_save_lock = threading.Lock()

# === Fungsi Kirim Telegram ===
def send_telegram_message(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    logging.info(f"Model Evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae
    
def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, n_estimators: int = 500, learning_rate: float = 0.05, early_stopping_rounds: Optional[int] = 50, random_state: int = 42) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_train, y_train)
    
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, n_estimators: int = 500, learning_rate: float = 0.05, early_stopping_rounds: Optional[int] = 50, random_state: int = 42) -> XGBRegressor:
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_train, y_train)
    
    return model

# === Training LSTM ===
def train_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    epochs: int = 55,
    batch_size: int = 32,
    verbose: int = 1
) -> Sequential:
    X_arr = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_arr, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# === Ambil & Validasi Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        # Gunakan 60 hari jika pakai interval 1 jam
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1h")

        required_cols = ["High", "Low", "Close", "Volume"]
        if df is not None and not df.empty and all(col in df.columns for col in required_cols) and len(df) >= 200:
            df["ticker"] = ticker
            return df

        logging.warning(f"{ticker}: Data kosong/kurang atau kolom tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def get_latest_close(ticker: str) -> float:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1d")
        if df.empty:
            logging.warning(f"{ticker}: Data kosong.")
            return None
        return df["Close"].iloc[-1]
    except Exception as e:
        logging.error(f"{ticker}: Gagal ambil harga terbaru - {e}")
        return None

# === Hitung Indikator ===
def calculate_indicators(df):
    # Pastikan index datetime
    df.index = pd.to_datetime(df.index)
    
    # Tambahkan fitur waktu
    HOURS_PER_DAY = 7
    df['hour'] = df.index.tz_convert('Asia/Jakarta').hour
    df['minute'] = df.index.tz_convert('Asia/Jakarta').minute

    df['is_opening_time'] = ((df['hour'] == 9) & (df['minute'] == 0)).astype(int)
    df['is_closing_time'] = ((df['hour'] == 15) & (df['minute'] == 49)).astype(int)

    # === Trend & Volatility Indicators ===
    df = ta.trend.adx(df['high'], df['low'], df['close'], window=14, fillna=True)
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # === Momentum Indicators ===
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['roc'] = ta.momentum.ROCIndicator(df['close'], window=14).roc()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=14).cci()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # === Volume Indicators ===
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
    df['efi'] = ta.volume.EaseOfMovementIndicator(df['high'], df['low'], df['volume'], window=14).ease_of_movement()
    df['nvi'] = ta.volume.NegativeVolumeIndexIndicator(df['close'], df['volume']).negative_volume_index()
    df['pvi'] = ta.volume.PositiveVolumeIndexIndicator(df['close'], df['volume']).positive_volume_index()

    # === Volatility: BBANDS, Keltner, Donchian ===
    bb = ta.volatility.BollingerBands(df['close'], window=14, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()

    kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=14)
    df['kc_upper'] = kc.keltner_channel_hband()
    df['kc_lower'] = kc.keltner_channel_lband()

    donchian = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=14)
    df['donchian_upper'] = donchian.donchian_channel_hband()
    df['donchian_lower'] = donchian.donchian_channel_lband()

    # === Moving Averages ===
    ma_periods = [10, 20, 50, 100, 200]
    for p in ma_periods:
        df[f'ma_{p}'] = ta.trend.SMAIndicator(df['close'], window=p).sma_indicator()
        df[f'ema_{p}'] = ta.trend.EMAIndicator(df['close'], window=p).ema_indicator()

    # === Linear Regression ===
    df['linreg'] = ta.trend.LinearRegressionIndicator(df['close'], window=14).linear_regression()

    # === Daily Aggregates ===
    df["daily_avg"] = df["close"].rolling(HOURS_PER_DAY).mean()
    df["daily_std"] = df["close"].rolling(HOURS_PER_DAY).std()
    df["daily_range"] = df["high"].rolling(HOURS_PER_DAY).max() - df["low"].rolling(HOURS_PER_DAY).min()

    # === Target prediksi: harga tertinggi & terendah BESOK ===
    df["future_high"] = df["high"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).max()
    df["future_low"]  = df["low"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).min()

    return df.dropna()

# Konstanta threshold (letakkan di atas fungsi analyze_stock)
MIN_PRICE = 500
MAX_PRICE = 2000
MIN_VOLUME = 10000
MIN_VOLATILITY = 0.005
MIN_PROB = 0.9

def is_stock_eligible(price, avg_volume, atr, ticker):
    if price < MIN_PRICE:
        logging.info(f"{ticker} dilewati: harga terlalu rendah ({price:.2f})")
        return False
    if price > MAX_PRICE:
        logging.info(f"{ticker} dilewati: harga terlalu tinggi ({price:.2f})")
        return False
    if avg_volume < MIN_VOLUME:
        logging.info(f"{ticker} dilewati: volume terlalu rendah ({avg_volume:.0f})")
        return False
    if (atr / price) < MIN_VOLATILITY:
        logging.info(f"{ticker} dilewati: volatilitas terlalu rendah (ATR={atr:.4f})")
        return False
    return True

def is_stock_eligible(price, avg_volume, atr, ticker) -> bool:
    # Tentukan kriteria kelayakan saham
    return price > 100 and avg_volume > 1000000 and atr > 1.5

# === Feature Preparation ===
def prepare_features_and_labels(df, features):
    df = df.dropna(subset=features + ["future_high", "future_low"])
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]
    return train_test_split(X, y_high, y_low, test_size=0.2, random_state=42)

# === Model Loading/Training ===
def load_or_train_model(path, train_fn, X_train, y_train, model_type="joblib"):
    if os.path.exists(path):
        model = joblib.load(path) if model_type == "joblib" else load_model(path)
        logging.info(f"Loaded model from {path}")
    else:
        model = train_fn(X_train, y_train)
        with model_save_lock:
            if model_type == "joblib":
                joblib.dump(model, path)
            else:
                model.save(path)
        logging.info(f"Trained & saved model to {path}")
    return model

# === Evaluation & Accuracy ===
def evaluate_prediction_accuracy() -> Dict[str, float]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File prediksi_log.csv tidak ditemukan.")
        return {}

    try:
        df_log = pd.read_csv(log_path, names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
    except Exception as e:
        logging.error(f"Gagal membaca file log prediksi: {e}")
        return {}

    df_data = get_realized_price_data()
    if df_data.empty:
        logging.warning("Data realisasi harga kosong.")
        return {}

    df_data["tanggal"] = pd.to_datetime(df_data["tanggal"])

    df_merged = df_log.merge(df_data, on=["ticker", "tanggal"], how="inner")

    if df_merged.empty:
        logging.info("Tidak ada prediksi yang cocok dengan data realisasi.")
        return {}

    df_merged["benar"] = (
        (df_merged["actual_high"] >= df_merged["pred_high"]) &
        (df_merged["actual_low"]  <= df_merged["pred_low"])
    )

    akurasi_per_ticker = df_merged.groupby("ticker")["benar"].mean().to_dict()
    logging.info(f"Akurasi prediksi dihitung untuk {len(akurasi_per_ticker)} ticker.")

    return akurasi_per_ticker

# === Fitur dan target ===
target = df["future_high"]
features = df.drop(columns=["future_high", "future_low"])  # Hindari data leakage

# === Split data ===
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

def make_objective_lgb(X_train, y_train, X_val, y_val):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)
    return objective

study = optuna.create_study(direction="minimize")
study.optimize(make_objective_lgb(X_train, y_train, X_val, y_val), n_trials=50)
best_lgb = lgb.LGBMRegressor(**study.best_params)
best_lgb.fit(X_train, y_train)

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=50)
best_xgb = XGBRegressor(**study.best_params)
best_xgb.fit(X_train, y_train)

def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units1', 32, 128, step=32), return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units2', 32, 128, step=32)))
    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='lstm_tuning',
    project_name='lstm'
)
X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
tuner.search(X_reshaped, y, epochs=30, validation_split=0.2, batch_size=32)
best_lstm = tuner.get_best_models(num_models=1)[0]

# === Reset Models ===
def reset_models():
    # Pola file model
    patterns = [
        "model_high_lgb_*.pkl",  # LightGBM
        "model_low_lgb_*.pkl",   # LightGBM
        "model_high_xgb_*.pkl",  # XGBoost
        "model_low_xgb_*.pkl",   # XGBoost
        "model_lstm_*.keras"     # LSTM
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
        logging.info("Tidak ada model yang ditemukan untuk dihapus.")
    else:
        logging.info(f"Total {total_deleted} model dihapus.")

# === Hitung Probabilitas Arah Prediksi ===
def calculate_probability(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    if "Close" not in X.columns:
        raise ValueError("'Close' column is required in input features (X).")
    if len(X) != len(y_true):
        raise ValueError("Length of X and y_true must match.")

    y_pred = model.predict(X)
    y_pred_series = pd.Series(y_pred, index=X.index)
    close_price = X["Close"]

    correct_dir = ((y_pred_series > close_price) & (y_true > close_price)) | \
                  ((y_pred_series < close_price) & (y_true < close_price))
    correct_dir = correct_dir.dropna()

    if len(correct_dir) == 0:
        return 0.0

    return correct_dir.sum() / len(correct_dir)
    
# === Main Analysis Function ===
def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return None

    df = calculate_indicators(df)
    # Validasi data
    if "ATR" not in df.columns or df["ATR"].dropna().empty:
        logging.warning(f"{ticker}: ATR kosong setelah kalkulasi.")
        return None

    atr = df["ATR"].dropna().iloc[-1]
    price = get_latest_close(ticker)
    if price is None:
        return None

    avg_volume = df["Volume"].tail(20).mean()

    if not is_stock_eligible(price, avg_volume, atr, ticker):
        logging.debug(f"{ticker}: Tidak memenuhi kriteria awal.")
        return None

    features = [
        # === Time-based features ===
        "hour", "is_opening_time", "is_closing_time",
        "daily_avg", "daily_std", "daily_range",

        # === Trend & Volatility Indicators ===
        "adx", "adx_pos", "adx_neg", "atr",

        # === Momentum Indicators ===
        "rsi", "stoch_k", "stoch_d", "williams_r", "roc",
        "mfi", "cci", "macd", "macd_signal", "macd_diff",

        # === Volume Indicators ===
        "obv", "cmf", "efi", "nvi", "pvi",

        # === Volatility Bands ===
        "bb_upper", "bb_middle", "bb_lower",
        "kc_upper", "kc_lower",
        "donchian_upper", "donchian_lower",

        # === Moving Averages ===
        "ma_10", "ema_10",
        "ma_20", "ema_20",
        "ma_50", "ema_50",
        "ma_100", "ema_100",
        "ma_200", "ema_200",

        # === Linear Regression ===
        "linreg"
    ]
    
    try:
        X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = prepare_features_and_labels(df, features)
    except Exception as e:
        logging.error(f"{ticker}: Error saat mempersiapkan data - {e}")
        return None

    # Latih dan muat model
    model_high_lgb = load_or_train_model(f"model_high_lgb_{ticker}.pkl", train_lightgbm, X_tr, yh_tr)
    model_low_lgb = load_or_train_model(f"model_low_lgb_{ticker}.pkl", train_lightgbm, X_tr, yl_tr)
    model_high_xgb = load_or_train_model(f"model_high_xgb_{ticker}.pkl", train_xgboost, X_tr, yh_tr)
    model_low_xgb = load_or_train_model(f"model_low_xgb_{ticker}.pkl", train_xgboost, X_tr, yl_tr)
    model_lstm = load_or_train_model(f"model_lstm_{ticker}.keras", train_lstm, X_tr, yh_tr, model_type="keras")

    # Hitung probabilitas
    prob_high_lgb = calculate_probability(model_high_lgb, X_te, yh_te)
    prob_low_lgb = calculate_probability(model_low_lgb, X_te, yl_te)
    prob_high_xgb = calculate_probability(model_high_xgb, X_te, yh_te)
    prob_low_xgb = calculate_probability(model_low_xgb, X_te, yl_te)

    # Gabungkan probabilitas
    prob_high = (prob_high_lgb + prob_high_xgb) / 2
    prob_low = (prob_low_lgb + prob_low_xgb) / 2

    if prob_high < MIN_PROB or prob_low < MIN_PROB:
        logging.info(f"{ticker} dilewati: Prob rendah (H={prob_high:.2f}, L={prob_low:.2f})")
        return None

    # Prediksi harga
    X_last = df[features].iloc[[-1]]
    ph_lgb = model_high_lgb.predict(X_last)[0]
    pl_lgb = model_low_lgb.predict(X_last)[0]
    ph_xgb = model_high_xgb.predict(X_last)[0]
    pl_xgb = model_low_xgb.predict(X_last)[0]
    ph_lstm = model_lstm.predict(np.reshape(X_last.values, (X_last.shape[0], X_last.shape[1], 1)))[0][0]
    pl_lstm = model_lstm.predict(np.reshape(X_last.values, (X_last.shape[0], X_last.shape[1], 1)))[0][0]

    # Ambil rata-rata prediksi
    ph = (ph_lgb + ph_xgb + ph_lstm) / 3
    pl = (pl_lgb + pl_xgb + pl_lstm) / 3

    # Tentukan aksi dan potensi profit
    action = "beli" if (ph - price) / price > 0.02 else "jual"
    profit_potential_pct = (ph - price) / price * 100 if action == "beli" else (price - pl) / price * 100

    if profit_potential_pct < 3:
        logging.info(f"{ticker} dilewati: potensi profit rendah ({profit_potential_pct:.2f}%)")
        return None

    result = {
        "ticker": ticker,
        "harga": price,
        "take_profit": ph,
        "stop_loss": pl,
        "profit_potential_pct": profit_potential_pct,
        "prob_success": (prob_high + prob_low) / 2,
        "aksi": action
    }

    logging.info(f"{ticker}: {action.upper()} | TP: {ph:.2f} | SL: {pl:.2f} | Potensi Profit: {profit_potential_pct:.2f}% | Probabilitas: {result['prob_success']*100:.1f}%")
    return result

def save_model(model, model_name: str, model_type: str = "sklearn", output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, model_name)
    
    if model_type == "sklearn":
        joblib.dump(model, filepath + ".pkl")
        logging.info(f"Model saved to {filepath}.pkl")
    elif model_type == "keras":
        model.save(filepath + ".keras")
        logging.info(f"Model saved to {filepath}.keras")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
# === Retrain Model If Accuracy is Low ===
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
            # === Time-based features ===
            "hour", "is_opening_time", "is_closing_time",
            "daily_avg", "daily_std", "daily_range",

            # === Trend & Volatility Indicators ===
            "adx", "adx_pos", "adx_neg", "atr",

            # === Momentum Indicators ===
            "rsi", "stoch_k", "stoch_d", "williams_r", "roc",
            "mfi", "cci", "macd", "macd_signal", "macd_diff",

            # === Volume Indicators ===
            "obv", "cmf", "efi", "nvi", "pvi",

            # === Volatility Bands ===
            "bb_upper", "bb_middle", "bb_lower",
            "kc_upper", "kc_lower",
            "donchian_upper", "donchian_lower",

            # === Moving Averages ===
            "ma_10", "ema_10",
            "ma_20", "ema_20",
            "ma_50", "ema_50",
            "ma_100", "ema_100",
            "ma_200", "ema_200",

            # === Linear Regression ===
            "linreg"
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

# === Motivation Quotes ===
MOTIVATION_QUOTES = [
    "Success is not the key to happiness. Happiness is the key to success.",
    "The only way to do great work is to love what you do.",
    "Don’t watch the clock; do what it does. Keep going.",
    "Success is the sum of small efforts, repeated day in and day out.",
    "The harder you work for something, the greater you’ll feel when you achieve it."
]

def get_random_motivation() -> str:
    return random.choice(MOTIVATION_QUOTES)

# === Main Execution & Signal Sending ===
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
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        send_telegram_message(message)

    logging.info("✅ Selesai.")
