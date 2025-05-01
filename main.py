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
import xgboost as xgb
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf

from xgboost import XGBRegressor
from keras_tuner import RandomSearch
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
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

# === Konfigurasi Bot ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7
BACKUP_CSV_PATH  = "stock_data_backup.csv"
HASH_PATH = "features_hash.json"
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

# === Ambil & Validasi Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        # Gunakan 60 hari jika pakai interval 1 jam
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1h")

        required_cols = ["high", "low", "close", "volume"]
        if df is not None and not df.empty and all(col in df.columns for col in required_cols) and len(df) >= 200:
            df["ticker"] = ticker
            return df

        logging.warning(f"{ticker}: Data kosong/kurang atau kolom tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
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
    max_trials: int = 10,
    executions_per_trial: int = 1,
    epochs: int = 20,
    batch_size: int = 32,
    verbose: int = 1,
    ticker: str = "default"
) -> Sequential:
    X_arr = np.reshape(X.values, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int("lstm_units_1", 32, 256, step=32),
            return_sequences=True,
            input_shape=(X_arr.shape[1], 1)
        ))
        model.add(Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1)))
        model.add(LSTM(
            units=hp.Int("lstm_units_2", 32, 256, step=32)
        ))
        model.add(Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1)))
        model.add(Dense(
            units=hp.Int("dense_units", 16, 128, step=16),
            activation="relu"
        ))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    tuner = RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="lstm_tuning",
        project_name=f"price_pred_{ticker}",
        overwrite=True
    )

    tuner.search(X_arr, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=verbose)

    # Simpan model terbaik
    best_model = tuner.get_best_models(1)[0]
    model_path = f"model_lstm_{ticker}.keras"
    best_model.save(model_path)

    # Simpan hasil tuning ke CSV
    results_df = pd.DataFrame(tuner.results_summary(num_trials=max_trials, print_results=False))
    results_csv_path = f"lstm_tuning_results_{ticker}.csv"
    with open(results_csv_path, "w") as f:
        tuner.results_summary(print_fn=lambda x: f.write(x + "\n"))

    return best_model

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    logging.info(f"Cross-validation scores: {scores}")
    
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

# Fungsi load_or_train_model
# === Fungsi Utama Load or Train Model ===
def load_or_train_model(path, train_func, X, y, model_type="joblib"):
    if os.path.exists(path):
        model = joblib.load(path) if model_type == "joblib" else tf.keras.models.load_model(path)
        logging.info(f"Loaded model from {path}")
    else:
        model = train_func(X, y)
        with model_save_lock:
            if model_type == "joblib":
                joblib.dump(model, path)
            else:
                model.save(path)
        logging.info(f"Trained & saved model to {path}")
    return model

# === Hyperparameter Tuning untuk XGBoost ===
def tune_xgboost_hyperparameters(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best XGBoost Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# === Hyperparameter Tuning untuk LightGBM ===
def tune_lightgbm_hyperparameters(X_train, y_train):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }
    grid_search = GridSearchCV(lgb.LGBMRegressor(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best LightGBM Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# === Fungsi Kirim Alert ===
def send_alert(message):
    logging.error(f"ALERT: {message}")

# === Fungsi Simpan dan Load Preprocessed Data ===
def save_preprocessed_data(df, path):
    df.to_csv(path, index=False)
    logging.info(f"Data diproses dan disimpan ke {path}")

def load_preprocessed_data(path):
    return pd.read_csv(path)

# === Fungsi Save Model dengan Versioning ===
def save_model_with_versioning(model, path):
    versioned_path = f"{path}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    joblib.dump(model, versioned_path)
    logging.info(f"Model disimpan dengan versi: {versioned_path}")

def get_feature_hash(features: list[str]) -> str:
    features_str = ",".join(sorted(features))
    return hashlib.md5(features_str.encode()).hexdigest()

def check_and_reset_model_if_needed(ticker: str, current_features: list[str]):
    current_hash = get_feature_hash(current_features)

    try:
        with open(HASH_PATH, "r") as f:
            saved_hashes = json.load(f)
    except FileNotFoundError:
        saved_hashes = {}

    if saved_hashes.get(ticker) != current_hash:
        logging.info(f"{ticker}: Struktur fitur berubah — melakukan reset model")

        # Hapus model XGBoost jika ada
        for suffix in ["high", "low"]:
            model_path_xgb = f"model_xgb_{suffix}_{ticker}.pkl"
            if os.path.exists(model_path_xgb):
                os.remove(model_path_xgb)
                logging.info(f"{ticker}: Model XGBoost '{suffix}' dihapus")

        # Hapus model LightGBM jika ada
        for suffix in ["high", "low"]:
            model_path_lgb = f"model_lgb_{suffix}_{ticker}.pkl"
            if os.path.exists(model_path_lgb):
                os.remove(model_path_lgb)
                logging.info(f"{ticker}: Model LightGBM '{suffix}' dihapus")

        # Hapus model LSTM jika ada
        lstm_path = f"model_lstm_{ticker}.keras"
        if os.path.exists(lstm_path):
            os.remove(lstm_path)
            logging.info(f"{ticker}: Model LSTM dihapus")

        # Simpan hash baru
        saved_hashes[ticker] = current_hash
        with open(HASH_PATH, "w") as f:
            json.dump(saved_hashes, f, indent=2)

        logging.info(f"{ticker}: Model akan dilatih ulang dengan struktur fitur terbaru")
    else:
        logging.debug(f"{ticker}: Struktur fitur sama — model tidak di-reset")
        
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

def prepare_features_and_labels(df, features):
    df = df.dropna(subset=features + ["future_high", "future_low"])
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]
    return train_test_split(X, y_high, y_low, test_size=0.2, random_state=42)

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

def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return None
    else:
        df.columns = [col.lower() for col in df.columns]

    df.columns = df.columns.str.lower()
    
    df = calculate_indicators(df)

    # Pastikan kolom dan nilainya ada sebelum lanjut
    if "ATR" not in df.columns or df["ATR"].dropna().empty:
        logging.warning(f"{ticker}: ATR kosong setelah kalkulasi.")
        return None

    atr = df["ATR"].dropna().iloc[-1]
    required_columns = ["high", "low", "close", "volume", "atr"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"{ticker}: Kolom yang diperlukan tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
        return None

    # Gunakan harga terbaru dari daily
    price = get_latest_close(ticker)

    # Fallback hanya jika df tidak kosong
    if price is None:
        if df is not None and not df.empty and "Close" in df.columns:
            price = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else None
        else:
            logging.warning(f"{ticker}: Data fallback juga kosong.")
            return None

    # Hentikan jika harga tetap tidak bisa didapatkan
    if price is None:
        logging.warning(f"{ticker}: Tidak bisa mendapatkan harga terbaru.")
        return None

    avg_volume = df["Volume"].tail(20).mean()
    atr = df["ATR"].iloc[-1]

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
    check_and_reset_model_if_needed(ticker, features)

    try:
        X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = prepare_features_and_labels(df, features)
    except Exception as e:
        logging.error(f"{ticker}: Error saat mempersiapkan data - {e}")
        return None

    # Latih dan muat model LightGBM dan XGBoost
    model_high_lgb = load_or_train_model(f"model_high_lgb_{ticker}.pkl", train_lightgbm, X_tr, yh_tr)
    model_low_lgb  = load_or_train_model(f"model_low_lgb_{ticker}.pkl", train_lightgbm, X_tr, yl_tr)
    model_high_xgb = load_or_train_model(f"model_high_xgb_{ticker}.pkl", train_xgboost, X_tr, yh_tr)
    model_low_xgb  = load_or_train_model(f"model_low_xgb_{ticker}.pkl", train_xgboost, X_tr, yl_tr)

    # Latih dan muat model LSTM
    model_lstm = load_or_train_model(f"model_lstm_{ticker}.keras", train_lstm, X_tr, yh_tr, model_type="keras")

    try:
        # Hitung probabilitas dari model LightGBM dan XGBoost
        prob_high_lgb = calculate_probability(model_high_lgb, X_te, yh_te)
        prob_low_lgb  = calculate_probability(model_low_lgb,  X_te, yl_te)
        prob_high_xgb = calculate_probability(model_high_xgb, X_te, yh_te)
        prob_low_xgb  = calculate_probability(model_low_xgb,  X_te, yl_te)
    except Exception as e:
        logging.error(f"{ticker}: Error saat menghitung probabilitas - {e}")
        return None

    # Gabungkan probabilitas dari kedua model (LightGBM dan XGBoost)
    prob_high = (prob_high_lgb + prob_high_xgb) / 2
    prob_low  = (prob_low_lgb + prob_low_xgb) / 2

    if prob_high < MIN_PROB or prob_low < MIN_PROB:
        logging.info(f"{ticker} dilewati: Prob rendah (H={prob_high:.2f}, L={prob_low:.2f})")
        return None

    # Ambil data terakhir untuk prediksi harga
    X_last = df[features].iloc[[-1]]
    
    # Prediksi harga dari LightGBM dan XGBoost
    ph_lgb = model_high_lgb.predict(X_last)[0]
    pl_lgb = model_low_lgb.predict(X_last)[0]
    ph_xgb = model_high_xgb.predict(X_last)[0]
    pl_xgb = model_low_xgb.predict(X_last)[0]

    # Prediksi harga dari LSTM
    ph_lstm = model_lstm.predict(np.reshape(X_last.values, (X_last.shape[0], X_last.shape[1], 1)))[0][0]
    pl_lstm = model_lstm.predict(np.reshape(X_last.values, (X_last.shape[0], X_last.shape[1], 1)))[0][0]

    # Ambil rata-rata dari hasil prediksi
    ph = (ph_lgb + ph_xgb + ph_lstm) / 3
    pl = (pl_lgb + pl_xgb + pl_lstm) / 3

    action = "beli" if (ph - price) / price > 0.02 else "jual"
    profit_potential_pct = (ph - price) / price * 100 if action == "beli" else (price - pl) / price * 100
    prob_succ = (prob_high + prob_low) / 2
    
    # Validasi sederhana agar TP dan SL masuk akal
    if action == "beli":
        if ph <= price or pl >= price:
            return None
    else:  # aksi == "jual"
        if ph >= price or pl <= price:
            return None
            
    if profit_potential_pct < 3:
        logging.info(f"{ticker} dilewati: potensi profit rendah ({profit_potential_pct:.2f}%)")
        return None

    tanggal = pd.Timestamp.now().strftime("%Y-%m-%d")
    log_prediction(ticker, tanggal, ph, pl, price)

    return {
        "ticker": ticker,
        "harga": round(price, 2),
        "take_profit": round(ph, 2),
        "stop_loss": round(pl, 2),
        "aksi": action,
        "prob_high": round(prob_high, 2),
        "prob_low": round(prob_low, 2),
        "prob_success": round(prob_succ, 2),
        "profit_potential_pct": round(profit_potential_pct, 2),
    }

def main():
    results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))
    
    # Urutkan berdasarkan potensi profit tertinggi
    results = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)

    # Ambil Top N
    top_n = 5  # atau 1 kalau mau satu sinyal terbaik saja
    top_signals = results[:top_n]

    for r in top_signals:
        print_signal(r)
        
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
        
def get_realized_price_data() -> pd.DataFrame:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        return pd.DataFrame()

    df_log = pd.read_csv(log_path)
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"])
    results = []

    for ticker in df_log["ticker"].unique():
        df_ticker = df_log[df_log["ticker"] == ticker].copy()
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
            print(f"Gagal download data untuk {ticker}: {e}")
            continue

        if df_price.empty or "High" not in df_price or "Low" not in df_price:
            print(f"Data kosong atau kolom hilang untuk {ticker}")
            continue

        df_price.index = pd.to_datetime(df_price.index)  # pastikan datetime
        df_price = df_price.sort_index()

        for _, row in df_ticker.iterrows():
            tanggal_prediksi = row["tanggal"]
            start_window = tanggal_prediksi + pd.Timedelta(days=1)
            end_window = tanggal_prediksi + pd.Timedelta(days=6)

            df_window = df_price.loc[(df_price.index >= start_window) & (df_price.index <= end_window)]
            if df_window.shape[0] < 3:
                continue

            results.append({
                "ticker": ticker,
                "tanggal": tanggal_prediksi,
                "actual_high": float(df_window["High"].max()),
                "actual_low": float(df_window["Low"].min())
            })

    return pd.DataFrame(results)
    
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
    
def check_and_reset_model_if_needed(ticker, features):
    hash_path = f"model_feature_hashes.json"
    
    # Hitung hash untuk fitur saat ini
    current_hash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()

    saved_hashes = {}
    if os.path.exists(hash_path):
        try:
            with open(hash_path, "r") as f:
                content = f.read().strip()
                if content:
                    saved_hashes = json.loads(content)
        except json.JSONDecodeError:
            logging.warning("Hash file corrupted, resetting...")
            saved_hashes = {}

    # Jika hash fitur berbeda, reset model
    if saved_hashes.get(ticker) != current_hash:
        logging.info(f"Fitur berubah untuk {ticker}, reset model.")
        
        # Hapus model-model yang sudah ada
        model_files = [
            f"model_high_lgb_{ticker}.pkl",  # LightGBM
            f"model_low_lgb_{ticker}.pkl",   # LightGBM
            f"model_high_xgb_{ticker}.pkl",  # XGBoost
            f"model_low_xgb_{ticker}.pkl",   # XGBoost
            f"model_lstm_{ticker}.keras"     # LSTM
        ]
        
        for fname in model_files:
            if os.path.exists(fname):
                os.remove(fname)
                logging.info(f"Model {fname} dihapus.")
        
        # Simpan hash baru untuk fitur
        saved_hashes[ticker] = current_hash
        with open(hash_path, "w") as f:
            json.dump(saved_hashes, f, indent=2)
    else:
        logging.info(f"Fitur untuk {ticker} tidak berubah, model tidak di-reset.")

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
        
# === Daftar Kutipan Motivasi ===
MOTIVATION_QUOTES = [
    "Setiap peluang adalah langkah kecil menuju kebebasan finansial.",
    "Cuan bukan tentang keberuntungan, tapi tentang konsistensi dan strategi.",
    "Disiplin hari ini, hasil luar biasa nanti.",
    "Trader sukses bukan yang selalu benar, tapi yang selalu siap.",
    "Naik turun harga itu biasa, yang penting arah portofolio naik.",
    "Fokus pada proses, profit akan menyusul.",
    "Jangan hanya lihat harga, lihat potensi di baliknya.",
    "Ketika orang ragu, itulah peluang sesungguhnya muncul.",
    "Investasi terbaik adalah pada pengetahuan dan ketenangan diri.",
    "Satu langkah hari ini lebih baik dari seribu penyesalan besok."
    "Moal rugi jalma nu talek jeung tekadna kuat.",
    "Rejeki mah moal ka tukang, asal usaha jeung sabar.",
    "Lamun hayang hasil nu beda, ulah make cara nu sarua terus.",
    "Ulah sieun gagal, sieun lamun teu nyobaan.",
    "Cuan nu leres asal ti élmu jeung kasabaran.",
    "Sabada hujan pasti aya panonpoé, sabada rugi bisa aya untung.",
    "Ngabagéakeun resiko teh bagian tina kamajuan.",
    "Jalma nu kuat téh lain nu teu pernah rugi, tapi nu sanggup bangkit deui.",
    "Ngora kudu wani nyoba, heubeul kudu wani investasi.",
    "Reureujeungan ayeuna, kabagjaan engké."
    "Niat alus, usaha terus, hasil bakal nuturkeun.",
    "Ulah ngadagoan waktu nu pas, tapi cobian ayeuna.",
    "Hirup teh kawas saham, kadang naek kadang turun, tapi ulah leungit arah.",
    "Sakumaha gede ruginya, élmu nu diala leuwih mahal hargana.",
    "Ulah beuki loba mikir, beuki saeutik tindakan.",
    "Kabagjaan datang ti tangtungan jeung harepan nu dilaksanakeun.",
    "Panghasilan teu datang ti ngalamun, tapi ti aksi jeung analisa.",
    "Sasat nu bener, bakal mawa kana untung nu lila.",
    "Tong ukur ningali batur nu untung, tapi diajar kumaha cara maranéhna usaha.",
    "Jalma sukses mah sok narima gagal minangka bagian ti perjalanan."
    "Saham bisa turun, tapi semangat kudu tetap ngora. Jalan terus, rejeki moal salah alamat.",
    "Kadang market galak, tapi inget, nu sabar jeung konsisten nu bakal panén hasilna.",
    "Cuan moal datang ti harepan hungkul, kudu dibarengan ku strategi jeung tekad.",
    "Teu aya jalan pintas ka sukses, ngan aya jalan nu jelas jeung disiplin nu kuat.",
    "Di balik koreksi aya akumulasi, di balik gagal aya élmu anyar. Ulah pundung!",
    "Sakumaha seredna pasar, nu kuat haténa bakal salamet.",
    "Rejeki teu datang ti candaan, tapi ti candak kaputusan jeung tindakan.",
    "Sugan ayeuna can untung, tapi tong hilap, tiap analisa téh tabungan pangalaman.",
    "Tenang lain berarti nyerah, tapi ngatur posisi jeung nunggu waktu nu pas.",
    "Sagalana dimimitian ku niat, dilaksanakeun ku disiplin, jeung dipanen ku waktu."
    "“Suatu saat akan datang hari di mana semua akan menjadi kenangan.” – Erza Scarlet (Fairy Tail)",
    "“Lebih baik menerima kejujuran yang pahit, daripada kebohongan yang manis.” – Soichiro Yagami (Death Note)",
    "“Jangan menyerah. Hal memalukan bukanlah ketika kau jatuh, tetapi ketika kau tidak mau bangkit lagi.” – Midorima Shintarou (Kuroko no Basuke)",
    "“Jangan khawatirkan apa yang dipikirkan orang lain. Tegakkan kepalamu dan melangkahlah ke depan.” – Izuku Midoriya (Boku no Hero Academia)",
    "“Tuhan tak akan menempatkan kita di sini melalui derita demi derita bila Ia tak yakin kita bisa melaluinya.” – Kano Yuki (Sword Art Online)",
    "“Mula-mula, kau harus mengubah dirimu sendiri atau tidak akan ada yang berubah untukmu.” – Sakata Gintoki (Gintama)",
    "“Banyak orang gagal karena mereka tidak memahami usaha yang diperlukan untuk menjadi sukses.” – Yukino Yukinoshita (Oregairu)",
    "“Kekuatan sejati dari umat manusia adalah bahwa kita memiliki kuasa penuh untuk mengubah diri kita sendiri.” – Saitama (One Punch Man)",
    "“Hidup bukanlah permainan keberuntungan. Jika kau ingin menang, kau harus bekerja keras.” – Sora (No Game No Life)",
    "“Kita harus mensyukuri apa yang kita punya saat ini karena mungkin orang lain belum tentu mempunyainya.” – Kayaba Akihiko (Sword Art Online)",
    "“Kalau kau ingin menangis karena gagal, berlatihlah lebih keras lagi sehingga kau pantas menangis ketika kau gagal.” – Megumi Takani (Samurai X)",
    "“Ketika kau bekerja keras dan gagal, penyesalan itu akan cepat berlalu. Berbeda dengan penyesalan ketika tidak berani mencoba.” – Akihiko Usami (Junjou Romantica)",
    "“Ketakutan bukanlah kejahatan. Itu memberitahukan apa kelemahanmu. Dan begitu tahu kelemahanmu, kamu bisa menjadi lebih kuat.” – Gildarts (Fairy Tail)",
    "“Untuk mendapatkan kesuksesan, keberanianmu harus lebih besar daripada ketakutanmu.” – Han Juno (Eureka Seven)",
    "“Kegagalan seorang pria yang paling sulit yaitu ketika dia gagal untuk menghentikan air mata seorang wanita.” – Kasuka Heiwajima (Durarara!)",
    "“Air mata palsu bisa menyakiti orang lain. Tapi, senyuman palsu hanya akan menyakiti dirimu sendiri.” – C.C (Code Geass)",
    "“Kita harus menjalani hidup kita sepenuhnya. Kamu tidak pernah tahu, kita mungkin sudah mati besok.” – Kaori Miyazono (Shigatsu wa Kimi no Uso)",
    "“Bagaimana kamu bisa bergerak maju kalau kamu terus menyesali masa lalu?” – Edward Elric (Fullmetal Alchemist: Brotherhood)",
    "“Jika kau seorang pria, buatlah wanita yang kau cintai jatuh cinta denganmu apa pun yang terjadi!” – Akhio (Clannad)",
    "“Semua laki-laki mudah cemburu dan bego, tapi perempuan malah menyukainya. Orang jadi bodoh saat jatuh cinta.” – Horo (Spice and Wolf)",
    "“Wanita itu sangat indah, satu senyuman mereka saja sudah menjadi sebuah keajaiban.” – Onigiri (Air Gear)",
    "“Saat kamu harus memilih satu cinta aja, pasti ada orang lain yang menangis.” – Tsubame (Ai Kora)",
    "“Aku tidak suka hubungan yang tidak jelas.” – Senjougahara (Bakemonogatari)",
    "“Cewek itu seharusnya lembut dan baik, dan bisa menyembuhkan luka di hati.” – Yoshii (Baka to Test)",
    "“Keluargamu adalah pahlawanmu.” – Sinchan (C. Sinchan)"
    "Hidup itu sederhana, kita yang membuatnya sulit. – Confucius.",
    "Hal yang paling penting adalah menikmati hidupmu, menjadi bahagia, apa pun yang terjadi. - Audrey Hepburn.",
    "Hidup itu bukan soal menemukan diri Anda sendiri, hidup itu membuat diri Anda sendiri. - George Bernard Shaw.",
    "Hidup adalah mimpi bagi mereka yang bijaksana, permainan bagi mereka yang bodoh, komedi bagi mereka yang kaya, dan tragedi bagi mereka yang miskin. - Sholom Aleichem.",
    "Kenyataannya, Anda tidak tahu apa yang akan terjadi besok. Hidup adalah pengendaraan yang gila dan tidak ada yang menjaminnya. – Eminem.",
    "Tujuan hidup kita adalah menjadi bahagia. - Dalai Lama.",
    "Hidup yang baik adalah hidup yang diinspirasi oleh cinta dan dipandu oleh ilmu pengetahuan. - Bertrand Russell.",
    "Seribu orang tua bisa bermimpi, satu orang pemuda bisa mengubah dunia. – Soekarno.",
    "Pendidikan adalah senjata paling ampuh untuk mengubah dunia. - Nelson Mandela.",
    "Usaha dan keberanian tidak cukup tanpa tujuan dan arah perencanaan. - John F. Kennedy.",
    "Dunia ini cukup untuk memenuhi kebutuhan manusia, bukan untuk memenuhi keserakahan manusia. - Mahatma Gandhi.",
    "Jika kamu berpikir terlalu kecil untuk membuat sebuah perubahan, cobalah tidur di ruangan dengan seekor nyamuk. - Dalai Lama.",
    "Anda mungkin bisa menunda, tapi waktu tidak akan menunggu. - Benjamin Franklin.",
    "Kamu tidak perlu menjadi luar biasa untuk memulai, tapi kamu harus memulai untuk menjadi luar biasa. - Zig Ziglar.",
    "Jangan habiskan waktumu memukuli dinding dan berharap bisa mengubahnya menjadi pintu. - Coco Chanel.",
    "Tidak ada yang akan berhasil kecuali kau melakukannya. - Maya Angelou.",
    "Kamu tidak bisa kembali dan mengubah awal saat kamu memulainya, tapi kamu bisa memulainya lagi dari mana kamu berada sekarang dan ubah akhirnya. - C.S Lewis.",
    "Beberapa orang memimpikan kesuksesan, sementara yang lain bangun setiap pagi untuk mewujudkannya. - Wayne Huizenga.",
    "Pekerjaan-pekerjaan kecil yang selesai dilakukan lebih baik daripada rencana-rencana besar yang hanya didiskusikan. - Peter Marshall.",
    "Kita harus berarti untuk diri kita sendiri dulu sebelum kita menjadi orang yang berharga bagi orang lain. - Ralph Waldo Emerson.",
    "Hal yang paling menyakitkan adalah kehilangan jati dirimu saat engkau terlalu mencintai seseorang. Serta lupa bahwa sebenarnya engkau juga spesial. - Ernest Hemingway.",
    "Beberapa orang akan pergi dari hidupmu, tapi itu bukan akhir dari ceritamu. Itu cuma akhir dari bagian mereka di ceritamu. - Faraaz Kazi.",
    "Cinta terjadi begitu singkat, namun melupakan memakan waktu begitu lama. - Pablo Neruda.",
    "Seseorang tak akan pernah tahu betapa dalam kadar cintanya sampai terjadi sebuah perpisahan. - Kahlil Gibran.",
    "Hubungan asmara itu seperti kaca. Terkadang lebih baik meninggalkannya dalam keadaan pecah daripada menyakiti dirimu dengan cara menyatukan mereka kembali. - D.Love.",
    "Cinta itu seperti angin. Kau tak dapat melihatnya, tapi kau dapat merasakannya. - Nicholas Sparks.",
    "Cinta adalah ketika kebahagiaan orang lain lebih penting dari kebahagiaanmu. - H. Jackson Brown.",
    "Asmara bukan hanya sekadar saling memandang satu sama lain. Tapi, juga sama-sama melihat ke satu arah yang sama. - Antoine de Saint-Exupéry.",
    "Bagaimana kau mengeja ‘cinta’? tanya Piglet. Kau tak usah mengejanya, rasakan saja, jawab Pooh. - A.A Milne.",
    "Kehidupan adalah 10 persen apa yang terjadi terhadap Anda dan 90 persen adalah bagaimana Anda meresponnya. - Lou Holtz.",
    "Satu-satunya keterbatasan dalam hidup adalah perilaku yang buruk. - Scott Hamilton.",
    "Seseorang yang berani membuang satu jam waktunya tidak mengetahui nilai dari kehidupan. - Charles Darwin.",
    "Apa yang kita pikirkan menentukan apa yang akan terjadi pada kita. Jadi jika kita ingin mengubah hidup, kita perlu sedikit mengubah pikiran kita. - Wayne Dyer.",
    "Ia yang mengerjakan lebih dari apa yang dibayar pada suatu saat nanti akan dibayar lebih dari apa yang ia kerjakan. - Napoleon Hill.",
    "Saya selalu mencoba untuk mengubah kemalangan menjadi kesempatan. - John D. Rockefeller.",
    "Seseorang yang pernah melakukan kesalahan dan tidak pernah memperbaikinya berarti ia telah melakukan satu kesalahan lagi. - Konfusius.",
    "Anda tidak akan pernah belajar sabar dan berani jika di dunia ini hanya ada kebahagiaan. - Helen Keller.",
    "Tidak apa-apa untuk merayakan kesuksesan, tapi lebih penting untuk memperhatikan pelajaran tentang kegagalan. – Bill Gates."
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
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        send_telegram_message(message)

    logging.info("✅ Selesai.")
