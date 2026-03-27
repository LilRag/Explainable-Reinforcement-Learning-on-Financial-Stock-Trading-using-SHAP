import yfinance as yf
import pandas as pd
import os

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

OHLC     = ["Open", "High", "Low", "Close"]
RAW_CSV  = "stock_data_raw.csv"
START    = "2014-01-01"
END      = "2022-06-21"


# ── Download ──────────────────────────────────────────────────────────────────
def download_raw(save_path=RAW_CSV):
    if os.path.exists(save_path):
        print(f"[skip] {save_path} already exists — delete it to re-download.")
        return

    print(f"Downloading {START} → {END} for {len(TICKERS)} tickers...")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=True
    )

    # Parse (ticker, price) MultiIndex → flat CSV with Ticker column
    frames = []
    missing = []

    for ticker in TICKERS:
        try:
            df = raw[ticker][OHLC].copy()
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if not df.empty:
                df["Ticker"] = ticker
                frames.append(df)
                print(f"  [ok] {ticker} — {len(df)} rows")
            else:
                missing.append(ticker)
                print(f"  [empty] {ticker}")
        except KeyError:
            missing.append(ticker)
            print(f"  [missing] {ticker}")

    if frames:
        combined = pd.concat(frames).sort_index()
        combined.to_csv(save_path)
        print(f"\nSaved {save_path} — {len(combined)} total rows, {len(frames)}/60 tickers")
    else:
        print("\n[ERROR] Nothing downloaded — still rate limited, try again later.")

    if missing:
        print(f"Missing tickers: {missing}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_raw()
    