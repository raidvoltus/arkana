# data/fetcher.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, interval="1h", lookback_days=30):
    """
    Mengambil data historis saham dari Yahoo Finance.
    
    Args:
        ticker (str): Kode saham.
        interval (str): Interval data (default: '1h').
        lookback_days (int): Jumlah hari ke belakang untuk pengambilan data.

    Returns:
        pd.DataFrame: Data historis saham.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    if df.empty or len(df) < 20:
        return pd.DataFrame()

    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    return df
