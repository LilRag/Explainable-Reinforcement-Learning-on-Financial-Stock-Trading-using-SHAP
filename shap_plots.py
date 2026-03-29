import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Re-run pipeline ────────────────────────────────────────────────────────────
import importlib.util
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = os.path.dirname(os.path.abspath(__file__))
ds   = load_module("data_splitting", os.path.join(base, "data_splitting.py"))
dqn  = load_module("dqn_trading",   os.path.join(base, "dqn_trading.py"))

train_data, test_data = ds.train_data, ds.test_data

TICKERS   = ["RELIANCE.BO", "TCS.BO", "AAPL", "MSFT"]
available = [t for t in TICKERS if t in train_data and t in test_data]
print(f"Running pipeline on: {available}")
results = dqn.run_pipeline(train_data, test_data, tickers_to_run=available)

os.makedirs("plots", exist_ok=True)

# colours matching the paper
BUY_COL  = "#E05C5C"   # red  — Buy day
SELL_COL = "#5C8DE0"   # blue — Sell day


# ─────────────────────────────────────────────────────────────────────────────
# WATERFALL PLOT (paper's main output)
# One bar per past day.
# Bar colour: red if that day's action was Buy, blue if Sell.
# Bar length: SHAP value for that day (contribution to reward prediction).
# ─────────────────────────────────────────────────────────────────────────────
def waterfall_plot(day_shap_row, day_actions_row, day_names, ticker, step_idx, save_path):
    """
    day_shap_row    : (30,) SHAP per day for this step
    day_actions_row : (30,) action (0 or 1) on each of those 30 past days
    """
    n      = len(day_shap_row)        # 30
    sv     = day_shap_row.copy()
    base   = 0.0
    # Running cumulative for waterfall starts
    running = base
    starts, widths, colours, labels = [], [], [], []

    for d in range(n):
        v = sv[d]
        starts.append(running if v >= 0 else running + v)
        widths.append(abs(v))
        # Colour by action on that day (paper: red=buy, blue=sell)
        colours.append(BUY_COL if day_actions_row[d] == 1 else SELL_COL)
        labels.append(day_names[d])
        running += v

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos   = np.arange(n)

    bars = ax.barh(y_pos, widths, left=starts, color=colours,
                   edgecolor="white", linewidth=0.4, height=0.7)

    # Value labels
    for bar, v in zip(bars, sv):
        x = bar.get_x() + bar.get_width() + max(abs(sv)) * 0.01
        if abs(v) > 0:
            ax.text(x, bar.get_y() + bar.get_height()/2,
                    f"{v:+.4f}", va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(base, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("SHAP value  (contribution to reward prediction)", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall Plot — {ticker}  |  Test Step {step_idx}",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)

    buy_patch  = mpatches.Patch(color=BUY_COL,  label="Buy day  (action = 1)")
    sell_patch = mpatches.Patch(color=SELL_COL, label="Sell day (action = 0)")
    ax.legend(handles=[buy_patch, sell_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BAR PLOT — mean |SHAP| per day across all test steps
# ─────────────────────────────────────────────────────────────────────────────
def bar_plot(mean_day_shap, day_names, ticker, save_path):
    idx  = np.argsort(mean_day_shap)   # ascending for horizontal bar
    vals = mean_day_shap[idx]
    lbls = [day_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colours = [BUY_COL if v > np.median(vals) else SELL_COL for v in vals]
    ax.barh(range(len(vals)), vals, color=colours, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(lbls, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|  (average influence on reward prediction)", fontsize=10)
    ax.set_title(f"Feature Importance (All Test Steps) — {ticker}",
                 fontsize=11, fontweight="bold")
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVE
# ─────────────────────────────────────────────────────────────────────────────
def training_curve(ep_rewards, ticker, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep_rewards, color=SELL_COL, linewidth=1.2, alpha=0.5, label="Episode reward")
    w    = max(1, len(ep_rewards)//10)
    roll = np.convolve(ep_rewards, np.ones(w)/w, mode="valid")
    ax.plot(range(w-1, len(ep_rewards)), roll,
            color=BUY_COL, linewidth=2, label=f"Rolling mean ({w} eps)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.set_title(f"DQN Training Curve — {ticker}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_facecolor("white")
    ax.grid(color="#EEEEEE", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Generate all plots
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
for ticker, r in results.items():
    print(f"\n[{ticker}]")
    safe = ticker.replace(".", "_")

    d_act  = r["day_actions"]   # (n_steps, 30)
    n_buys = d_act.sum(axis=1)  # how many buy days in each step's window
    best   = int(np.argmax(n_buys)) if n_buys.max() > 0 else 0

    waterfall_plot(
        r["day_shap"][best],
        r["day_actions"][best],
        r["day_names"],
        ticker, best,
        f"plots/{safe}_waterfall.png"
    )
    bar_plot(
        r["mean_day_shap"], r["day_names"],
        ticker, f"plots/{safe}_bar.png"
    )
    training_curve(
        r["ep_rewards"], ticker,
        f"plots/{safe}_training.png"
    )

print("\nAll plots saved to ./plots/")
for f in sorted(os.listdir("plots")):
    print(f"  plots/{f}")