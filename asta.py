# === Import: Standard Library ===
import os
import glob
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from logging.handlers import RotatingFileHandler

# === Import: External Libraries ===
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

from ta import momentum, trend, volatility, volume
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from concurrent.futures import ThreadPoolExecutor

# === Bot Configuration ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7  # in days
BACKUP_CSV_PATH  = "stock_data_backup.csv"
HASH_PATH        = "features_hash.json"

# === Stock List ===
STOCK_LIST = []