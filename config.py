# config.py

config = {
    # Daftar ticker saham yang akan dianalisis
    "TICKERS": ["BBCA.JK", "TLKM.JK", "BBRI.JK", "UNVR.JK"],

    # Panjang window data historis yang digunakan untuk pelatihan dan prediksi
    "WINDOW_SIZE": 60,  # 60 jam terakhir

    # Path penyimpanan model
    "MODEL_DIR": "models/saved_models",

    # Threshold minimal probabilitas prediksi yang digunakan untuk validasi sinyal
    "PROBABILITY_THRESHOLD": 0.6,

    # Estimasi potensi profit minimum untuk sinyal layak
    "MIN_PROFIT_PCT": 2.0,

    # Filter awal saham berdasarkan kriteria kelayakan
    "MIN_PRICE": 500,
    "MIN_VOLUME": 100000,
    "MIN_VOLATILITY": 0.01,  # 1%

    # Telegram settings
    "TELEGRAM_TOKEN": "YOUR_TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID": "YOUR_TELEGRAM_CHAT_ID",

    # Auto retrain settings
    "RETRAIN_THRESHOLD_MSE": 0.005,
    "MAX_DAYS_BEFORE_RETRAIN": 7,
}
