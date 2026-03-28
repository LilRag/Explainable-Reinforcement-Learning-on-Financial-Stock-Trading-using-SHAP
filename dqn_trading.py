"""
Explainable RL on Stock Trading using SHAP (DQN)
=================================================
Paper: https://arxiv.org/abs/2208.08790

Faithful implementation:
1. Environment   : gym-anytrading StocksEnv, window=30, actions Buy(1)/Sell(0)
2. DQN           : 2-hidden-layer MLP, experience replay, target network
3. SHAP features : past 30 days' [action, reward] pairs -> 60-dim vector
4. SHAP target   : reward prediction (not Q-value)
5. Output        : per-day SHAP contributions
"""

import numpy as np
import pandas as pd
import random
import warnings
from collections import deque
from gym_anytrading.envs import StocksEnv
import shap

warnings.filterwarnings("ignore")

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE   = 30
N_ACTIONS     = 2
HIDDEN1       = 50
HIDDEN2       = 50
LR            = 0.005
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE    = 32
MEMORY_SIZE   = 2000
EPISODES      = 30
SHAP_BG       = 50


# ── Normalise OHLC dataframe ──────────────────────────────────────────────────
def normalise_df(df):
    """
    Divide all OHLC columns by the first Close value so prices are
    in the range ~1x. This keeps gym-anytrading rewards small and
    stable, preventing gradient explosion in the MLP.
    The paper uses Yahoo Finance data directly; normalisation is the
    standard pre-processing step to make the env learnable.
    """
    scale = df["Close"].iloc[0]
    return (df / scale).astype(np.float32)


# ── Numpy MLP ─────────────────────────────────────────────────────────────────
class MLP:
    def __init__(self, in_dim, h1, h2, out_dim):
        self.W1 = np.random.randn(in_dim, h1) * np.sqrt(2 / in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, out_dim) * np.sqrt(2 / h2)
        self.b3 = np.zeros(out_dim)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = np.atleast_2d(x)
        self.z1 = x @ self.W1 + self.b1;    self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3

    def predict(self, x):
        return self.forward(x)

    def copy_from(self, other):
        for attr in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            setattr(self, attr, getattr(other, attr).copy())

    def update(self, x, tgt, lr):
        x = np.atleast_2d(x)
        q = self.forward(x)
        dL = (q - tgt) / x.shape[0]
        dW3 = self.a2.T @ dL;      db3 = dL.sum(0)
        da2 = dL @ self.W3.T;      dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2;     db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T;     dz1 = da1 * (self.z1 > 0)
        dW1 = x.T @ dz1;           db1 = dz1.sum(0)
        self.W3 -= lr * dW3;  self.b3 -= lr * db3
        self.W2 -= lr * dW2;  self.b2 -= lr * db2
        self.W1 -= lr * dW1;  self.b1 -= lr * db1


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim):
        self.epsilon = EPSILON_START
        self.memory  = deque(maxlen=MEMORY_SIZE)
        self.model   = MLP(state_dim, HIDDEN1, HIDDEN2, N_ACTIONS)
        self.target  = MLP(state_dim, HIDDEN1, HIDDEN2, N_ACTIONS)
        self.target.copy_from(self.model)
        self.steps   = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(N_ACTIONS)
        return int(np.argmax(self.model.predict(state)[0]))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        S  = np.array([b[0] for b in batch])
        A  = np.array([b[1] for b in batch])
        R  = np.array([b[2] for b in batch])
        S2 = np.array([b[3] for b in batch])
        D  = np.array([b[4] for b in batch])
        Q  = self.model.predict(S)
        Qn = self.target.predict(S2)
        T  = Q.copy()
        for i in range(BATCH_SIZE):
            T[i, A[i]] = R[i] if D[i] else R[i] + GAMMA * np.max(Qn[i])
        self.model.update(S, T, LR)
        self.steps += 1
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        if self.steps % 10 == 0:
            self.target.copy_from(self.model)


# ── gym-anytrading env ────────────────────────────────────────────────────────
def make_env(df):
    """
    Normalise df then wrap in StocksEnv.
    Normalisation ensures reward signal is in range ~[-0.1, 0.1]
    so the MLP can learn. Without it all rewards are 0 or huge.
    """
    norm = normalise_df(df)
    env  = StocksEnv(df=norm, window_size=WINDOW_SIZE,
                     frame_bound=(WINDOW_SIZE, len(norm)))
    obs, _ = env.reset()
    return env, obs.flatten().shape[0]


# ── Train ─────────────────────────────────────────────────────────────────────
def train_agent(ticker, train_df):
    env, state_dim = make_env(train_df)
    agent   = DQNAgent(state_dim=state_dim)
    ep_rews = []

    for ep in range(1, EPISODES + 1):
        obs, _  = env.reset()
        state   = obs.flatten()
        total   = 0.0
        done = trunc = False
        while not (done or trunc):
            action              = agent.act(state)
            obs2, r, done, trunc, _ = env.step(action)
            next_state          = obs2.flatten()
            agent.remember(state, action, r, next_state, done or trunc)
            agent.replay()
            state = next_state
            total += r
        ep_rews.append(total)
        if ep % 10 == 0:
            print(f"  ep {ep:3d}/{EPISODES}  reward={total:.4f}  eps={agent.epsilon:.3f}")

    return agent, ep_rews


# ── Test: collect actions + rewards ──────────────────────────────────────────
def collect_test_experience(test_df, agent):
    """Run trained agent on test set with epsilon=0 (greedy)."""
    agent.epsilon = 0.0
    env, _  = make_env(test_df)
    obs, _  = env.reset()
    state   = obs.flatten()
    actions, rewards = [], []
    done = trunc = False

    while not (done or trunc):
        action              = agent.act(state)
        obs2, r, done, trunc, _ = env.step(action)
        actions.append(action)
        rewards.append(float(r))
        state = obs2.flatten()

    print(f"  Steps={len(actions)}  Buy%={100*np.mean(actions):.1f}%  "
          f"Total reward={sum(rewards):.4f}")
    return np.array(actions, dtype=np.float32), \
           np.array(rewards, dtype=np.float32)


# ── Build SHAP feature matrix ─────────────────────────────────────────────────
def build_shap_features(actions, rewards):
    """
    Paper: "the sliding window of the past 30 days' actions and rewards
            is used as the feature vector"

    For each step t >= 30:
      X[t] = [action_{t-30}, reward_{t-30}, ..., action_{t-1}, reward_{t-1}]
             shape (60,)  — interleaved action, reward per day
      y[t] = reward at step t  (what SHAP explains)
    """
    X, y = [], []
    for t in range(WINDOW_SIZE, len(actions)):
        row = np.empty(WINDOW_SIZE * 2, dtype=np.float32)
        row[0::2] = actions[t - WINDOW_SIZE: t]
        row[1::2] = rewards[t - WINDOW_SIZE: t]
        X.append(row)
        y.append(rewards[t])
    return np.array(X), np.array(y)


# ── SHAP ──────────────────────────────────────────────────────────────────────
def explain_with_shap(X, y):
    """
    Paper: "SHAP calculates the contribution of each of those 30 days
            to the current reward prediction"

    Fits a linear reward predictor on the 60-dim window -> reward,
    then runs KernelExplainer on it.
    """
    X_b  = np.hstack([X, np.ones((len(X), 1))])
    coefs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

    def predict_reward(x):
        xb = np.hstack([np.atleast_2d(x),
                         np.ones((len(np.atleast_2d(x)), 1))])
        return xb @ coefs

    bg_idx     = np.random.choice(len(X), min(SHAP_BG, len(X)), replace=False)
    background = X[bg_idx]
    test_X     = X[:100]

    print(f"  Running SHAP (bg={len(background)}, explaining={len(test_X)} steps)...")
    explainer = shap.KernelExplainer(predict_reward, background)
    shap_vals = explainer.shap_values(test_X, nsamples=100, silent=True)
    return shap_vals, test_X, y[:100]


# ── Aggregate 60-dim SHAP to 30 per-day values ───────────────────────────────
def per_day_shap(shap_vals, actions, start_idx=WINDOW_SIZE):
    """
    Sum action-SHAP and reward-SHAP for each day into one value.
    Also record which action was taken that day (for waterfall colour).
    """
    n = shap_vals.shape[0]
    day_shap    = np.zeros((n, WINDOW_SIZE))
    day_actions = np.zeros((n, WINDOW_SIZE))
    for t in range(n):
        for d in range(WINDOW_SIZE):
            day_shap[t, d]    = shap_vals[t, d*2] + shap_vals[t, d*2 + 1]
            day_actions[t, d] = actions[start_idx + t - WINDOW_SIZE + d]
    return day_shap, day_actions


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

        print("Training DQN (gym-anytrading)...")
        agent, ep_rews = train_agent(ticker, train_data[ticker])
        print(f"  Final ep reward: {ep_rews[-1]:.4f}")

        print("Collecting test experience...")
        actions, rewards = collect_test_experience(test_data[ticker], agent)

        X, y = build_shap_features(actions, rewards)
        print(f"  SHAP feature matrix: {X.shape}")

        shap_vals, test_X, test_y = explain_with_shap(X, y)
        d_shap, d_actions         = per_day_shap(shap_vals, actions)
        mean_d_shap               = np.abs(d_shap).mean(axis=0)
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
            "day_actions"  : d_actions,
            "mean_day_shap": mean_d_shap,
            "day_names"    : dnames,
            "test_X"       : test_X,
            "test_y"       : test_y,
        }

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "data_splitting",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_splitting.py")
    )
    ds = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ds)
    train_data, test_data = ds.train_data, ds.test_data

    DEMO      = ["RELIANCE.BO", "TCS.BO", "AAPL", "MSFT"]
    available = [t for t in DEMO if t in train_data and t in test_data]
    print(f"Running on: {available}")
    results   = run_pipeline(train_data, test_data, tickers_to_run=available)

    print("\nSummary:")
    print(f"{'Ticker':<20} {'Test Reward':>12} {'Top Day':>15}")
    print("-" * 50)
    for ticker, r in results.items():
        top = r["day_names"][np.argmax(r["mean_day_shap"])]
        print(f"{ticker:<20} {r['rewards'].sum():>12.4f} {top:>15}")
