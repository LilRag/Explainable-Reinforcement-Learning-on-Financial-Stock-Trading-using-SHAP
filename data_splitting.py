import pandas as pd

OHLC = ["Open", "High", "Low", "Close"]

def load_data(path="stock_data_raw.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    train_raw = df[df.index <= "2021-05-31"]
    test_raw  = df[(df.index >= "2021-06-01") & (df.index <= "2022-06-21")]

    train_data = {t: grp.drop(columns="Ticker")
                  for t, grp in train_raw.groupby("Ticker")}
    test_data  = {t: grp.drop(columns="Ticker")
                  for t, grp in test_raw.groupby("Ticker")}

    print(f"Train: {len(train_data)} tickers | Test: {len(test_data)} tickers")
    return train_data, test_data

# module-level — imported by dqn_trading.py and visualisation.py
train_data, test_data = load_data()

# only save CSVs when run directly, not when imported
if __name__ == "__main__":
    pd.concat([df.assign(Ticker=t) for t, df in train_data.items()]).sort_index().to_csv("train_data.csv")
    pd.concat([df.assign(Ticker=t) for t, df in test_data.items()]).sort_index().to_csv("test_data.csv")
    print("Saved train_data.csv and test_data.csv")

    for ticker in ["RELIANCE.BO", "AAPL"]:
        t = train_data[ticker]
        v = test_data[ticker]
        print(f"{ticker} | train: {t.index.min().date()} → {t.index.max().date()} ({len(t)} rows) | test: {v.index.min().date()} → {v.index.max().date()} ({len(v)} rows)")