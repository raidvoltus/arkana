import os
import glob
import time
import threading
import joblib
import hashlib
import json
import requests
import random
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf

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

        required_cols = ["High", "Low", "Close", "Volume"]
        if df is not None and not df.empty and all(col in df.columns for col in required_cols) and len(df) >= 200:
            df["ticker"] = ticker
            return df

        logging.warning(f"{ticker}: Data kosong/kurang atau kolom tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

# === Hitung Indikator ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    HOURS_PER_DAY = 7
    HOURS_PER_WEEK = 35  # 5 hari trading, 7 jam per hari
    
    # Pastikan index sudah dalam timezone Asia/Jakarta
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jakarta")
    else:
        df.index = df.index.tz_convert("Asia/Jakarta")
        
    # === Indikator teknikal utama ===
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
    
    macd = trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()
    
    bb = volatility.BollingerBands(df["Close"], window=20)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    df["Support"] = df["Low"].rolling(window=48).min()
    df["Resistance"] = df["High"].rolling(window=48).max()

    df["RSI"] = momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["SMA_14"] = trend.SMAIndicator(df["Close"], window=14).sma_indicator()
    df["SMA_28"] = trend.SMAIndicator(df["Close"], window=28).sma_indicator()
    df["SMA_84"] = trend.SMAIndicator(df["Close"], window=84).sma_indicator()
    df["EMA_10"] = trend.EMAIndicator(df["Close"], window=10).ema_indicator()
    df["VWAP"] = volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()
    df["ADX"] = trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14).adx()
    df["CCI"] = trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()
    df["Momentum"] = momentum.ROCIndicator(df["Close"], window=12).roc()
    df["WilliamsR"] = momentum.WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=14).williams_r()

    # === Fitur waktu harian ===
    df["hour"] = df.index.hour
    df["is_opening_hour"] = (df["hour"] == 9).astype(int)
    df["is_closing_hour"] = (df["hour"] == 15).astype(int)
    df["daily_avg"] = df["Close"].rolling(HOURS_PER_DAY).mean()
    df["daily_std"] = df["Close"].rolling(HOURS_PER_DAY).std()
    df["daily_range"] = df["High"].rolling(HOURS_PER_DAY).max() - df["Low"].rolling(HOURS_PER_DAY).min()

    # === Target prediksi: harga tertinggi & terendah MINGGU DEPAN ===
    df["future_high"] = df["High"].shift(-HOURS_PER_WEEK).rolling(HOURS_PER_WEEK).max()
    df["future_low"]  = df["Low"].shift(-HOURS_PER_WEEK).rolling(HOURS_PER_WEEK).min()

    return df.dropna()