import pandas as pd
OHLC =["Open","High","Low","Close"]

def load_and_parse_raw_csv(filepath="stock_data.csv"):
    raw = pd.read_csv(filepath, header=None)

    # row 0 -> ticker names
    # row 1 -> price field names
    # row 2 -> 'date' label
    # row 3 onward = actual data 

    #extract dates from row 3
    dates = pd.to_datetime(raw.iloc[3:,0].values, format="mixed")


    ticker_row = raw.iloc[0,1:].values #skip first col ('ticker')
    price_row = raw.iloc[1,1:].values #skip first col ('date)


    tickers_found = ticker_row[::5].tolist()

    result ={}
    num_cols = raw.shape[1] -1  

    for i, ticker in enumerate(tickers_found):
        base_col = 1 + i*5
        o_col = base_col + 0 
        h_col = base_col + 1
        l_col = base_col + 2
        c_col = base_col + 3


        if c_col >= raw.shape[1]:
            print(f"Skipping {ticker} — columns out of range")
            continue

        # Extract OHLC data rows (skip first 3 header rows)
        ohlc = raw.iloc[3:, [o_col, h_col, l_col, c_col]].copy()
        ohlc.columns = OHLC
        ohlc.index   = dates
        ohlc = ohlc.apply(pd.to_numeric, errors="coerce").dropna()
        ohlc = ohlc.sort_index()

        if ohlc.empty:
            print(f"Warning: {ticker} has no data")
        else:
            result[ticker] = ohlc

    print(f"Parsed {len(result)} tickers")
    return result


# ── Load and split ────────────────────────────────────────────────────────────
all_data = load_and_parse_raw_csv("stock_data.csv")

# Split on date — no re-download needed
train_data = {t: df[df.index < "2021-06-01"]  for t, df in all_data.items()}
test_data  = {t: df[df.index >= "2021-06-01"] for t, df in all_data.items()}

# Drop tickers that ended up empty after split
train_data = {t: df for t, df in train_data.items() if not df.empty}
test_data  = {t: df for t, df in test_data.items()  if not df.empty}

print(f"Train: {len(train_data)} tickers")
print(f"Test:  {len(test_data)} tickers")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\nRELIANCE.BO train:")
print(train_data["RELIANCE.BO"].head())
print("\nAAPL test:")
print(test_data["AAPL"].head())