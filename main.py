# main.py

import os
import time
import traceback
from config import config
from data.fetcher import fetch_stock_data
from data.processor import prepare_features
from models.lightgbm_model import predict_with_lightgbm
from models.lstm_model import predict_with_lstm
from models.retrainer import auto_retrain_if_needed
from analysis.analyzer import analyze_stock
from telegram.notifier import send_signal
from data.monitor import evaluate_prediction_accuracy
from utils.logger import get_logger

logger = get_logger("main")

def process_ticker(ticker: str):
    try:
        logger.info(f"Processing ticker: {ticker}")
        
        # Step 1: Ambil data historis
        df = fetch_stock_data(ticker)

        # Step 2: Preprocessing dan feature engineering
        df_features = prepare_features(df)

        # Step 3: Prediksi dengan model LightGBM & LSTM
        lightgbm_preds = predict_with_lightgbm(ticker, df_features)
        lstm_preds = predict_with_lstm(ticker, df_features)

        # Step 4: Analisis sinyal berdasarkan prediksi dan filter
        signal = analyze_stock(ticker, df_features, lightgbm_preds, lstm_preds)

        if signal:
            # Step 5: Kirim sinyal ke Telegram
            send_signal(signal)

        # Step 6: Evaluasi dan auto-retrain jika perlu
        evaluate_prediction_accuracy(ticker)
        auto_retrain_if_needed(ticker, df_features)

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        traceback.print_exc()

def main():
    try:
        for ticker in config["TICKERS"]:
            analyze_and_signal(ticker)
    except Exception as e:
        send_telegram_message(f"Error saat menjalankan bot: {e}")

if __name__ == "__main__":
    main()
