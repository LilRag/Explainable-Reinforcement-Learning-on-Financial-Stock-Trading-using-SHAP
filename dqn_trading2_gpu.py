import numpy as np
import pandas as pd
import random
import warnings
from collections import deque
from gym_anytrading.envs import StocksEnv
import shap
import pickle
from sklearn.linear_model import Ridge

# ── TensorFlow import ─────────────────────────────────────────────────────────
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")


def configure_torch_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[gpu] Using {torch.cuda.get_device_name(device)}")
        return device

    print("[gpu-warning] PyTorch cannot see CUDA. This run will continue on CPU.")
    return torch.device("cpu")


TRAIN_DEVICE = configure_torch_device()

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE       = 30
N_ACTIONS         = 2
LR                = 1e-3

GAMMA             = 0.95
EPSILON_START     = 1.0
EPSILON_MIN_TRAIN = 0.05
EPSILON_MIN_TEST  = 0.0

# CHANGED: 30 → 10 episodes.
# Epsilon decay recalculated so exploration still collapses sensibly:
#   Old: 0.90^30 ≈ 0.04  (just hits floor at ep ~25)
#   New: 0.80^10 ≈ 0.11  (reaches ~0.11 by ep 10, floors at 0.05 by ep ~13)
# 0.80 per episode is more aggressive — the agent becomes greedy faster,
# which is fine because we have fewer episodes to learn from anyway.
EPSILON_DECAY     = 0.80   # was 0.90 for 30 eps

BATCH_SIZE        = 16
MEMORY_SIZE       = 1000
EPISODES          = 10     # was 30

# SHAP: reduced background + explain samples to match shorter test rollouts.
# With fewer episodes the agent collects fewer unique test steps, so
# SHAP_BG=30 (was 50) and SHAP_EXPLAIN=50 (was 100) keeps it fast + stable.
SHAP_BG           = 30     # was 50
SHAP_EXPLAIN      = 50     # was 100 (new explicit constant)
SHAP_TARGET_HORIZON = 5

TAU               = 0.005


# ── Normalise OHLC dataframe ──────────────────────────────────────────────────
def normalise_df(df):
    scale = df['Close'].iloc[0]
    return (df / scale).astype('float32')


class DenseRewardStocksEnv(StocksEnv):
    def _calculate_reward(self, action):
        current_price = self.prices[self._current_tick]
        previous_price = self.prices[self._current_tick - 1]
        price_diff = current_price - previous_price
        return float(price_diff if action == 1 else -price_diff)


# ── Build Keras model ─────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def build_torch_model(in_dim: int) -> QNetwork:
    return QNetwork(in_dim).to(TRAIN_DEVICE)


# ── Soft update helper ────────────────────────────────────────────────────────
def soft_update(online: nn.Module, target: nn.Module, tau: float = TAU):
    with torch.no_grad():
        for w_online, w_target in zip(online.parameters(), target.parameters()):
            w_target.data.mul_(1.0 - tau).add_(w_online.data, alpha=tau)


# ── DDQN Agent ────────────────────────────────────────────────────────────────
class DDQNAgent:
    def __init__(self, state_dim: int):
        self.epsilon = EPSILON_START
        self.memory  = deque(maxlen=MEMORY_SIZE)
        self.steps   = 0

        self.model  = build_torch_model(state_dim)
        self.target = build_torch_model(state_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.target.load_state_dict(self.model.state_dict())
        self.initial_weights = [p.detach().clone() for p in self.model.parameters()]

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(N_ACTIONS)
        self.model.eval()
        with torch.no_grad():
            state_t = torch.as_tensor(np.atleast_2d(state), dtype=torch.float32, device=TRAIN_DEVICE)
            q = self.model(state_t).detach().cpu().numpy()
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        S  = torch.as_tensor(np.array([b[0] for b in batch], dtype=np.float32), device=TRAIN_DEVICE)
        A  = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.long, device=TRAIN_DEVICE)
        R  = torch.as_tensor(np.array([b[2] for b in batch], dtype=np.float32), device=TRAIN_DEVICE)
        S2 = torch.as_tensor(np.array([b[3] for b in batch], dtype=np.float32), device=TRAIN_DEVICE)
        D  = torch.as_tensor(np.array([b[4] for b in batch], dtype=bool), device=TRAIN_DEVICE)

        self.model.train()
        Q_pred = self.model(S)

        with torch.no_grad():
            best_actions = torch.argmax(self.model(S2), dim=1)
            q_next_target = self.target(S2).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            next_values = R + GAMMA * q_next_target * (~D).float()
            target_q = Q_pred.detach().clone()
            target_q[torch.arange(BATCH_SIZE, device=TRAIN_DEVICE), A] = next_values

        loss_t = self.loss_fn(Q_pred, target_q)
        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()

        loss = float(loss_t.detach().cpu().item())
        Q = Q_pred.detach().cpu().numpy()

        if self.steps % 50 == 0:
            avg_q = np.mean(Q)
            print(f"[train] step={self.steps} loss={loss:.5f} avg_q={avg_q:.4f} eps={self.epsilon:.3f}")

        if self.steps % 200 == 0:
            diff = np.mean([
                torch.mean(torch.abs(w.detach() - iw)).item()
                for w, iw in zip(self.model.parameters(), self.initial_weights)
            ])
            print(f"[weights] avg change={diff:.6f}")

        self.steps += 1
        soft_update(self.model, self.target, TAU)

        return loss


# ── gym-anytrading env ────────────────────────────────────────────────────────
def make_env(df):
    norm = normalise_df(df)
    env  = DenseRewardStocksEnv(df=norm, window_size=WINDOW_SIZE, frame_bound=(WINDOW_SIZE, len(norm)))
    obs, _ = env.reset()
    return env, obs.flatten().shape[0]


# ── Train ─────────────────────────────────────────────────────────────────────
def train_agent(ticker, train_df):
    env, state_dim = make_env(train_df)
    agent    = DDQNAgent(state_dim=state_dim)
    ep_rews  = []

    for ep in range(1, EPISODES + 1):
        obs, _  = env.reset()
        state   = obs.flatten()
        total   = 0.0
        ep_losses = []
        done = trunc = False

        while not (done or trunc):
            action               = agent.act(state)
            obs2, r, done, trunc, _ = env.step(action)
            next_state           = obs2.flatten()
            agent.remember(state, action, r, next_state, done or trunc)

            loss = agent.replay()
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            total += r

        ep_rews.append(total)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0

        # More aggressive decay (0.80) so agent is reasonably greedy by ep 10
        agent.epsilon = max(EPSILON_MIN_TRAIN, agent.epsilon * EPSILON_DECAY)
        print(f"  Ep {ep:3d}/{EPISODES} | Reward: {total:>8.4f} | Avg Loss: {avg_loss:>8.5f} | Eps: {agent.epsilon:.3f}")

    return agent, ep_rews


# ── Test: collect actions + rewards ──────────────────────────────────────────
def collect_test_experience(test_df, agent):
    agent.epsilon = 0
    env, _ = make_env(test_df)
    obs, _ = env.reset()

    actions, rewards, states = [], [], []
    done = trunc = False

    while not (done or trunc):
        state = obs.flatten()
        states.append(state)

        action = agent.act(state)
        obs2, r, done, trunc, _ = env.step(action)

        actions.append(action)
        rewards.append(float(r))
        obs = obs2

    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    states = np.array(states, dtype=np.float32)
    switches = int(np.sum(actions[1:] != actions[:-1])) if len(actions) > 1 else 0
    nonzero = int(np.sum(np.abs(rewards) > 1e-9))
    print(
        f"  Steps={len(actions)}  Buy%={100*np.mean(actions):.1f}%  "
        f"Switches={switches}  Nonzero rewards={nonzero}/{len(rewards)}  "
        f"Reward std={np.std(rewards):.6f}  Total reward={sum(rewards):.4f}"
    )
    return actions, rewards, states


# ── Build SHAP feature matrix ─────────────────────────────────────────────────
# Sliding window of past 30 days' (action, reward) pairs.
# Shape: (n_steps, WINDOW_SIZE * 2) = (n, 60)
def build_shap_features(actions, rewards):
    X, y = [], []
    stop = len(actions) - SHAP_TARGET_HORIZON + 1
    for t in range(WINDOW_SIZE, stop):
        row = np.empty(WINDOW_SIZE * 2, dtype=np.float32)
        row[0::2] = actions[t - WINDOW_SIZE: t]
        row[1::2] = rewards[t - WINDOW_SIZE: t]
        X.append(row)
        y.append(np.sum(rewards[t: t + SHAP_TARGET_HORIZON]))
    return np.array(X, dtype=np.float32).reshape(-1, WINDOW_SIZE * 2), np.array(y, dtype=np.float32)


# ── SHAP on (action, reward) feature matrix ───────────────────────────────────
# FIX: previously used raw env states (~330 features) which caused per_day_shap
# to compute wrong n_cols and produce all-zero SHAP values.
#
# Now uses Ridge surrogate on (action, reward) window features (shape n, 60).
# SHAP_BG=30, SHAP_EXPLAIN=50 scaled down to match 10-episode shorter rollouts.
def explain_with_shap_features(shap_X: np.ndarray, shap_y: np.ndarray):
    n         = len(shap_X)
    bg_n      = min(SHAP_BG, n)
    explain_n = min(SHAP_EXPLAIN, n)

    if bg_n < 2:
        print("  [SHAP] Not enough test steps for SHAP — skipping.")
        dummy = np.zeros((explain_n, shap_X.shape[1]))
        return dummy, shap_X[:explain_n], shap_y[:explain_n]

    if np.allclose(shap_y, shap_y[0]):
        print("  [SHAP] Target rewards are constant; SHAP values will be zero.")

    surrogate = Ridge(alpha=1.0).fit(shap_X, shap_y)
    bg_idx    = np.random.choice(n, bg_n, replace=False)
    explainer = shap.KernelExplainer(surrogate.predict, shap_X[bg_idx])
    shap_vals = np.asarray(explainer.shap_values(shap_X[:explain_n], nsamples=100, silent=True))

    return shap_vals, shap_X[:explain_n], shap_y[:explain_n]


# ── Aggregate to per-day SHAP ─────────────────────────────────────────────────
# shap_vals: (n_steps, 60)  →  n_cols = 60 // 30 = 2  (action + reward per day)
def per_day_shap(shap_vals, actions, start_idx=WINDOW_SIZE):
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[:, :, 0]

    n_steps = shap_vals.shape[0]
    n_cols  = shap_vals.shape[1] // WINDOW_SIZE   # = 2
    day_shap    = np.zeros((n_steps, WINDOW_SIZE))
    day_abs_shap = np.zeros((n_steps, WINDOW_SIZE))
    day_actions = np.zeros((n_steps, WINDOW_SIZE))

    for t in range(n_steps):
        for d in range(WINDOW_SIZE):
            day_vals = shap_vals[t, d * n_cols: (d + 1) * n_cols]
            day_shap[t, d]     = day_vals.sum()
            day_abs_shap[t, d] = np.abs(day_vals).sum()
            day_actions[t, d] = actions[min(start_idx + t - WINDOW_SIZE + d,
                                            len(actions) - 1)]

    return day_shap, day_abs_shap, day_actions


def day_names():
    return [f"day_t-{WINDOW_SIZE - d}" for d in range(WINDOW_SIZE)]


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(train_data, test_data, tickers_to_run=None):
    if tickers_to_run is None:
        tickers_to_run = list(train_data.keys())

    results = {}
    for ticker in tickers_to_run:
        if ticker not in train_data or ticker not in test_data:
            print(f"[skip] {ticker}"); continue

        print(f"\n{'='*55}\nTicker: {ticker}\n{'='*55}")

        print(f"Training DDQN ({EPISODES} episodes, ε-decay={EPSILON_DECAY})...")
        agent, ep_rews = train_agent(ticker, train_data[ticker])
        print(f"  Final ep reward: {ep_rews[-1]:.4f}")

        print("Collecting test experience...")
        actions, rewards, states = collect_test_experience(test_data[ticker], agent)

        print(f"Computing SHAP values (bg={SHAP_BG}, explain={SHAP_EXPLAIN}, horizon={SHAP_TARGET_HORIZON})...")
        shap_X, shap_y            = build_shap_features(actions, rewards)
        shap_vals, test_X, test_y = explain_with_shap_features(shap_X, shap_y)
        d_shap, d_abs_shap, d_actions = per_day_shap(shap_vals, actions)
        mean_d_shap               = d_abs_shap.mean(axis=0)
        dnames                    = day_names()
        top5                      = np.argsort(mean_d_shap)[::-1][:5]

        print(f"\n  Top 5 most influential days:")
        for rank, idx in enumerate(top5, 1):
            print(f"    {rank}. {dnames[idx]:12s}  mean|SHAP|={mean_d_shap[idx]:.5f}")

        results[ticker] = {
            "agent"        : agent,
            "ep_rewards"   : ep_rews,
            "actions"      : actions,
            "rewards"      : rewards,
            "shap_values"  : shap_vals,
            "day_shap"     : d_shap,
            "day_abs_shap" : d_abs_shap,
            "day_actions"  : d_actions,
            "mean_day_shap": mean_d_shap,
            "day_names"    : dnames,
        }

    return results


# ── Diagnostic ────────────────────────────────────────────────────────────────
def diagnose_env(ticker, train_df, test_df):
    print(f"\n--- Diagnosing {ticker} ---")
    env, _ = make_env(train_df)

    def sample_rewards(pattern, steps=20):
        env.reset()
        sampled = []
        for i in range(steps):
            action = pattern[i % len(pattern)]
            _, r, done, trunc, _ = env.step(action)
            sampled.append(float(r))
            if done or trunc:
                break
        return sampled

    rewards_buy = sample_rewards([1])
    rewards_sell = sample_rewards([0])
    rewards_alt = sample_rewards([1, 0])

    norm_train = normalise_df(train_df)
    norm_test = normalise_df(test_df)
    train_bh = norm_train["Close"].iloc[-1] - norm_train["Close"].iloc[WINDOW_SIZE]
    test_bh = norm_test["Close"].iloc[-1] - norm_test["Close"].iloc[WINDOW_SIZE]

    print(f"  Sample BUY  rewards: {[round(x,4) for x in rewards_buy[:5]]}")
    print(f"  Sample SELL rewards: {[round(x,4) for x in rewards_sell[:5]]}")
    print(f"  Sample ALT  rewards: {[round(x,4) for x in rewards_alt[:5]]}")
    print(f"  Nonzero BUY/SELL/ALT: {sum(abs(x)>1e-9 for x in rewards_buy)}/{sum(abs(x)>1e-9 for x in rewards_sell)}/{sum(abs(x)>1e-9 for x in rewards_alt)}")
    print(f"  Normalized train Buy&Hold: {train_bh:.4f}")
    print(f"  Normalized test  Buy&Hold: {test_bh:.4f}")
    print(f"  Train df shape: {train_df.shape}")
    print(f"  Test  df shape: {test_df.shape}")
    print(f"  Train df columns: {train_df.columns.tolist()}")
    print(f"  Train df sample:\n{train_df.head(3)}")


# ── Save / load results ───────────────────────────────────────────────────────
def save_results(results, path="results.pkl"):
    slim = {}
    for ticker, r in results.items():
        slim[ticker] = {k: v for k, v in r.items() if k != "agent"}
    with open(path, "wb") as f:
        pickle.dump(slim, f)
    print(f"Results saved to {path}")

def load_results(path="results.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "data_splitting",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_splitting.py")
    )
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    train_data, test_data = ds.train_data, ds.test_data

    diagnose_env("RELIANCE.BO", train_data["RELIANCE.BO"], test_data["RELIANCE.BO"])
    diagnose_env("AAPL",        train_data["AAPL"],        test_data["AAPL"])

    available = list(train_data.keys())
    print(f"\nRunning DDQN on: {available}")
    demo_tickers = ["RELIANCE.BO", "TCS.BO", "AAPL", "MSFT"]
    results = run_pipeline(train_data, test_data, tickers_to_run=demo_tickers)

    print("\nSummary:")
    print(f"{'Ticker':<20} {'Test Reward':>12} {'Top Day':>15}")
    print("-" * 50)
    for ticker, r in results.items():
        top = r["day_names"][np.argmax(r["mean_day_shap"])]
        print(f"{ticker:<20} {r['rewards'].sum():>12.4f} {top:>15}")

    summary = []
    for ticker, r in results.items():
        summary.append({
            "ticker":       ticker,
            "index":        "SENSEX" if ".BO" in ticker else "DJIA",
            "total_reward": r["rewards"].sum(),
            "buy_pct":      round(100 * r["actions"].mean(), 2),
            "top_day":      r["day_names"][np.argmax(r["mean_day_shap"])],
            "top_shap":     round(r["mean_day_shap"].max(), 6),
        })
    pd.DataFrame(summary).to_csv("results_summary_ddqn.csv", index=False)

    print("\nDDQN vs Buy & Hold:")
    for ticker, r in results.items():
        norm = normalise_df(test_data[ticker])
        bh = norm["Close"].iloc[-1] - norm["Close"].iloc[WINDOW_SIZE]
        print(f"{ticker:20s} | DDQN: {r['rewards'].sum():8.4f} | Buy&Hold(norm): {bh:8.4f}")

    save_results(results)
    print("\nResults saved to results.pkl — run shap_plots.py to generate plots.")
