import os
import time
import pandas as pd
import yfinance as yf

TICKERS = [
    "ASIANPAINT.BO", "AXISBANK.BO", "BAJAJ-AUTO.BO", "BAJFINANCE.BO",
    "BAJAJFINSV.BO", "BHARTIARTL.BO", "HCLTECH.BO", "HDFCBANK.BO",
    "HINDUNILVR.BO", "ICICIBANK.BO", "INDUSINDBK.BO", "INFY.BO",
    "ITC.BO", "KOTAKBANK.BO", "LT.BO", "M&M.BO", "MARUTI.BO",
    "NESTLEIND.BO", "NTPC.BO", "ONGC.BO", "POWERGRID.BO",
    "RELIANCE.BO", "SBIN.BO", "SUNPHARMA.BO", "TATASTEEL.BO",
    "TCS.BO", "TECHM.BO", "TITAN.BO", "ULTRACEMCO.BO", "ADANIPORTS.BO",
    "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO",
    "KO", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD",
    "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ",
    "V", "WMT", "AMZN", "NVDA", "SHW"
]

OHLC       = ["Open", "High", "Low", "Close"]
TRAIN_CSV  = "stock_data_train.csv"
TEST_CSV   = "stock_data_test.csv"
TRAIN_START, TRAIN_END = "2014-01-01", "2021-05-31"
TEST_START,  TEST_END  = "2021-06-01", "2022-06-21"


# ── Downloader ────────────────────────────────────────────────────────────────
def download_to_csv(start, end, save_path, wait=0):
    """Single batched yfinance call → flat CSV. Skipped if file already exists."""
    if os.path.exists(save_path):
        print(f"[skip] {save_path} already exists — delete it to re-download.")
        return

    if wait:
        print(f"Waiting {wait}s before download...")
        time.sleep(wait)

    print(f"Downloading {start} → {end} ...")
    raw = yf.download(
        TICKERS, start=start, end=end,
        interval="1d", auto_adjust=True,
        group_by="ticker", progress=True
    )

    # Parse (ticker, price) MultiIndex → flat CSV with Ticker column
    frames = []
    for ticker in TICKERS:
        try:
            df = raw[ticker][OHLC].dropna()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if not df.empty:
                df = df.copy()
                df["Ticker"] = ticker
                frames.append(df)
        except KeyError:
            print(f"  [missing] {ticker}")

    if frames:
        pd.concat(frames).sort_index().to_csv(save_path)
        print(f"Saved {save_path} ({len(frames)} tickers)")
    else:
        print(f"[ERROR] Nothing to save — all tickers failed. Try again later.")


# ── Loader ────────────────────────────────────────────────────────────────────
def load_data():
    """
    Load train/test data from CSVs.
    If CSVs don't exist, downloads them first (one batched call each).
    Returns two dicts: {ticker: DataFrame(OHLC)}
    """
    download_to_csv(TRAIN_START, TRAIN_END, TRAIN_CSV)
    download_to_csv(TEST_START,  TEST_END,  TEST_CSV, wait=15)  # small gap between calls

    train_raw = pd.read_csv(TRAIN_CSV, index_col=0, parse_dates=True)
    test_raw  = pd.read_csv(TEST_CSV,  index_col=0, parse_dates=True)

    train_data = {t: grp.drop(columns="Ticker")
                  for t, grp in train_raw.groupby("Ticker")}
    test_data  = {t: grp.drop(columns="Ticker")
                  for t, grp in test_raw.groupby("Ticker")}

    print(f"Train: {len(train_data)} tickers | Test: {len(test_data)} tickers")
    return train_data, test_data


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_data, test_data = load_data()
    print("\nRELIANCE.BO train sample:")
    print(train_data["RELIANCE.BO"].head())
    print("\nAAPL test sample:")
    print(test_data["AAPL"].head())