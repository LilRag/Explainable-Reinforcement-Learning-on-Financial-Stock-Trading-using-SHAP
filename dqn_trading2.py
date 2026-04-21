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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

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

TAU               = 0.005


# ── Normalise OHLC dataframe ──────────────────────────────────────────────────
def normalise_df(df):
    scale = df['Close'].iloc[0]
    return (df / scale).astype('float32')


# ── Build Keras model ─────────────────────────────────────────────────────────
def build_keras_model(in_dim: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(in_dim,)),
        Dense(128, activation='relu'),
        Dense(64,  activation='relu'),
        Dense(N_ACTIONS, activation='linear'),
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


# ── Soft update helper ────────────────────────────────────────────────────────
def soft_update(online: tf.keras.Model, target: tf.keras.Model, tau: float = TAU):
    for w_online, w_target in zip(online.weights, target.weights):
        w_target.assign(tau * w_online + (1.0 - tau) * w_target)


# ── DDQN Agent ────────────────────────────────────────────────────────────────
class DDQNAgent:
    def __init__(self, state_dim: int):
        self.epsilon = EPSILON_START
        self.memory  = deque(maxlen=MEMORY_SIZE)
        self.steps   = 0

        self.model  = build_keras_model(state_dim)
        self.target = build_keras_model(state_dim)

        self.target.set_weights(self.model.get_weights())
        self.initial_weights = [w.copy() for w in self.model.get_weights()]

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(N_ACTIONS)
        q = self.model(np.atleast_2d(state), training=False).numpy()
        return int(np.argmax(q[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        S  = np.array([b[0] for b in batch], dtype=np.float32)
        A  = np.array([b[1] for b in batch])
        R  = np.array([b[2] for b in batch], dtype=np.float32)
        S2 = np.array([b[3] for b in batch], dtype=np.float32)
        D  = np.array([b[4] for b in batch], dtype=bool)

        Q = self.model(S, training=False).numpy()

        best_actions  = np.argmax(self.model(S2, training=False).numpy(), axis=1)
        Q_next_target = self.target(S2, training=False).numpy()

        T = Q.copy()
        for i in range(BATCH_SIZE):
            if D[i]:
                T[i, A[i]] = R[i]
            else:
                T[i, A[i]] = R[i] + GAMMA * Q_next_target[i, best_actions[i]]

        history = self.model.fit(S, T, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        loss    = history.history['loss'][0]

        if self.steps % 50 == 0:
            avg_q = np.mean(Q)
            print(f"[train] step={self.steps} loss={loss:.5f} avg_q={avg_q:.4f} eps={self.epsilon:.3f}")

        if self.steps % 200 == 0:
            diff = np.mean([
                np.mean(np.abs(w - iw))
                for w, iw in zip(self.model.get_weights(), self.initial_weights)
            ])
            print(f"[weights] avg change={diff:.6f}")

        self.steps += 1
        soft_update(self.model, self.target, TAU)

        return loss


# ── gym-anytrading env ────────────────────────────────────────────────────────
def make_env(df):
    norm = normalise_df(df)
    env  = StocksEnv(df=norm, window_size=WINDOW_SIZE, frame_bound=(WINDOW_SIZE, len(norm)))
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

    print(f"  Steps={len(actions)}  Buy%={100*np.mean(actions):.1f}%  Total reward={sum(rewards):.4f}")
    return np.array(actions), np.array(rewards), np.array(states)


# ── Build SHAP feature matrix ─────────────────────────────────────────────────
# Sliding window of past 30 days' (action, reward) pairs.
# Shape: (n_steps, WINDOW_SIZE * 2) = (n, 60)
def build_shap_features(actions, rewards):
    X, y = [], []
    for t in range(WINDOW_SIZE, len(actions)):
        row = np.empty(WINDOW_SIZE * 2, dtype=np.float32)
        row[0::2] = actions[t - WINDOW_SIZE: t]
        row[1::2] = rewards[t - WINDOW_SIZE: t]
        X.append(row)
        y.append(rewards[t])
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

        print(f"Computing SHAP values (bg={SHAP_BG}, explain={SHAP_EXPLAIN})...")
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
    env = StocksEnv(df=train_df, window_size=WINDOW_SIZE, frame_bound=(WINDOW_SIZE, len(train_df)))
    obs, _ = env.reset()

    rewards_buy, rewards_sell = [], []
    for _ in range(20):
        _, r, done, trunc, _ = env.step(1)
        rewards_buy.append(r)
        if done or trunc: break

    env.reset()
    for _ in range(20):
        _, r, done, trunc, _ = env.step(0)
        rewards_sell.append(r)
        if done or trunc: break

    print(f"  Sample BUY  rewards: {[round(x,4) for x in rewards_buy[:5]]}")
    print(f"  Sample SELL rewards: {[round(x,4) for x in rewards_sell[:5]]}")
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
        closes = test_data[ticker]["Close"].values
        bh     = closes[-1] - closes[0]
        print(f"{ticker:20s} | DDQN: {r['rewards'].sum():8.2f} | Buy&Hold: {bh:8.2f}")

    save_results(results)
    print("\nResults saved to results.pkl — run shap_plots.py to generate plots.")
