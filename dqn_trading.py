import numpy as np
import pandas as pd
import random
import warnings
from collections import deque
from gym_anytrading.envs import StocksEnv
import shap

# ── TensorFlow import ─────────────────────────────────────────────────────────
# CHANGED: replaces the hand-rolled numpy MLP.
# suppress noisy TF startup logs before import
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Most values kept identical to the original paper so the ablation is clean.
WINDOW_SIZE   = 30
N_ACTIONS     = 2
LR            = 1e-3          # CHANGED: 0.005 SGD → 1e-3 Adam (standard for Adam)
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_MIN   = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE    = 32
MEMORY_SIZE   = 2000
EPISODES      = 30
SHAP_BG       = 50
TAU           = 0.005         # CHANGED NEW: soft-update rate for target network


# ── Normalise OHLC dataframe ──────────────────────────────────────────────────
# UNCHANGED from original
def normalise_df(df):
    scale = df['Close'].iloc[0]
    return (df / scale).astype('float32')


# ── Build Keras model ─────────────────────────────────────────────────────────
# CHANGED: replaces the MLP class entirely.
#
# Architecture: in_dim → 128 → 64 → N_ACTIONS  (deeper than original 50→50
# because Adam handles larger nets stably — SGD would diverge here).
#
# Why not keep 50→50?  The original paper used a tiny net because manual
# SGD with LR=0.005 is unstable with more neurons. Adam's adaptive per-
# parameter learning rates let us scale up safely, giving the network more
# capacity to learn non-linear Q-value surfaces.
#
# compile() is called here once.  During replay we call model.fit() for a
# single gradient step — Keras handles backprop automatically.
def build_keras_model(in_dim: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(in_dim,)),
        Dense(128, activation='relu'),   # wider than original 50 neurons
        Dense(64,  activation='relu'),
        Dense(N_ACTIONS, activation='linear'),   # linear output = raw Q-values
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse')
    return model


# ── Soft update helper ────────────────────────────────────────────────────────
# CHANGED: replaces the hard copy_from() every 10 steps.
#
# Soft update: θ_target ← τ·θ_online + (1-τ)·θ_target
#
# Why soft instead of hard?
#   Hard copy creates a sudden shift in the Q-target every N steps, which
#   can cause periodic spikes in the loss. Soft update blends weights
#   continuously, giving smoother and more stable training — especially
#   important when the network is larger and has more parameters to track.
#
# τ = 0.005 means the target moves 0.5% toward the online network each step.
def soft_update(online: tf.keras.Model, target: tf.keras.Model, tau: float = TAU):
    for w_online, w_target in zip(online.weights, target.weights):
        w_target.assign(tau * w_online + (1.0 - tau) * w_target)


# ── DDQN Agent ────────────────────────────────────────────────────────────────
# CHANGED: was DQNAgent with numpy MLP. Now uses Keras + Adam + DDQN target.
#
# The two structural changes vs the original DQNAgent:
#   1. model / target are Keras models, not MLP instances
#   2. replay() computes DDQN targets instead of plain DQN targets
#      (see the detailed comment inside replay())
class DDQNAgent:
    def __init__(self, state_dim: int):
        self.epsilon = EPSILON_START
        self.memory  = deque(maxlen=MEMORY_SIZE)
        self.steps   = 0

        # Two separate networks — identical architecture, independent weights.
        # online:  receives gradients, used to SELECT actions
        # target:  frozen each step, used to EVALUATE selected actions
        self.model  = build_keras_model(state_dim)   # online network
        self.target = build_keras_model(state_dim)   # target network

        # Initialise target weights == online weights so training starts stable
        self.target.set_weights(self.model.get_weights())
        self.initial_weights = [w.copy() for w in self.model.get_weights()]
        

    # ── Action selection (unchanged interface) ────────────────────────────────
    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(N_ACTIONS)
        q = self.model(np.atleast_2d(state), training=False).numpy()
        return int(np.argmax(q[0]))

    # ── Memory (unchanged) ────────────────────────────────────────────────────
    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    # ── DDQN replay ──────────────────────────────────────────────────────────
    # This is the core algorithmic change.
    #
    # Plain DQN target (original):
    #   T = R + γ · max_a Q_target(S', a)
    #
    # Problem: using the same network to both SELECT and EVALUATE the best
    # next action consistently overestimates Q-values. Noisy Q-values get
    # picked more often (because argmax favours noise), and then their
    # inflated values propagate back through training. This is called
    # "maximisation bias" and causes the agent to be overly optimistic,
    # especially early in training when Q-values are inaccurate.
    #
    # DDQN fix (van Hasselt et al., 2016):
    #   a* = argmax_a  Q_online(S', a)   ← ONLINE selects the action
    #   T  = R + γ · Q_target(S', a*)    ← TARGET evaluates that action
    #
    # The online network (noisier, more up-to-date) picks what it thinks
    # is the best action. The target network (smoother, older) gives a
    # more conservative value estimate for that specific action.
    # Decoupling selection from evaluation eliminates the upward bias.
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        S  = np.array([b[0] for b in batch], dtype=np.float32)
        A  = np.array([b[1] for b in batch])
        R  = np.array([b[2] for b in batch], dtype=np.float32)
        S2 = np.array([b[3] for b in batch], dtype=np.float32)
        D  = np.array([b[4] for b in batch], dtype=bool)

        # Current Q-values from online network (becomes our update target)
        Q = self.model(S, training=False).numpy()

        # ── DDQN target computation ──────────────────────────────────────────
        # Step 1: online network decides which action is best in each S'
        best_actions = np.argmax(
            self.model(S2, training=False).numpy(),
            axis=1
        )   # shape: (BATCH_SIZE,)

        # Step 2: target network evaluates Q-value of that specific action
        Q_next_target = self.target(S2, training=False).numpy()
                        # shape: (BATCH_SIZE, N_ACTIONS)

        # Step 3: build the target vector, only updating the taken action's slot
        T = Q.copy()
        for i in range(BATCH_SIZE):
            if D[i]:
                # Terminal state: no future reward
                T[i, A[i]] = R[i]
            else:
                # DDQN: target evaluates the action online chose, not its own best
                T[i, A[i]] = R[i] + GAMMA * Q_next_target[i, best_actions[i]]

        # Single gradient step on the online network via Keras
        # verbose=0 silences per-batch logs

        history = self.model.fit(S, T, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.steps % 50 == 0:
            avg_q = np.mean(Q)
            print(f"[train] step={self.steps} loss={loss:.5f} avg_q={avg_q:.4f} eps={self.epsilon:.3f}")

        if self.steps % 200 == 0:
            diff = np.mean([
                np.mean(np.abs(w - iw))
                for w, iw in zip(self.model.get_weights(), self.initial_weights)
            ])
            print(f"[weights] avg change={diff:.6f}")

        # Epsilon decay (unchanged)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            
        history = self.model.fit(S, T, batch_size=BATCH_SIZE, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Soft-update target weights every step (replaces hard copy every 10)
        self.steps += 1
        soft_update(self.model, self.target, TAU)


# ── gym-anytrading env ────────────────────────────────────────────────────────
# UNCHANGED — environment and observation shape are identical
def make_env(df):
    norm = normalise_df(df)
    env  = StocksEnv(df=norm, window_size=WINDOW_SIZE, frame_bound=(WINDOW_SIZE, len(norm)))
    obs, _ = env.reset()
    return env, obs.flatten().shape[0]


# ── Train ─────────────────────────────────────────────────────────────────────
# CHANGED: DQNAgent → DDQNAgent. Loop body is otherwise identical.
def train_agent(ticker, train_df):
    env, state_dim = make_env(train_df)
    agent    = DDQNAgent(state_dim=state_dim)
    ep_rews  = []

    for ep in range(1, EPISODES + 1):
        obs, _  = env.reset()
        state   = obs.flatten()
        total   = 0.0
        ep_losses = [] # Track losses for this episode
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
        
        # Print stats for EVERY episode
        print(f"  Ep {ep:3d}/{EPISODES} | Reward: {total:>8.4f} | Avg Loss: {avg_loss:>8.5f} | Eps: {agent.epsilon:.3f}")

    return agent, ep_rews


# ── Test: collect actions + rewards ──────────────────────────────────────────
# UNCHANGED — epsilon=0 disables exploration during test
def collect_test_experience(test_df, agent):
    agent.epsilon = 0
    env, _ = make_env(test_df)
    obs, _ = env.reset()
    state  = obs.flatten()
    actions, rewards = [], []
    done = trunc = False
    while not (done or trunc):
        action               = agent.act(state)
        obs2, r, done, trunc, _ = env.step(action)
        actions.append(action)
        rewards.append(float(r))
        state = obs2.flatten()

    agent.epsilon = EPSILON_START

    print(f"  Steps={len(actions)}  Buy%={100*np.mean(actions):.1f}%  "
          f"Total reward={sum(rewards):.4f}")
    return np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32)


# ── Build SHAP feature matrix ─────────────────────────────────────────────────
# UNCHANGED — same sliding window of past 30 days' (action, reward) pairs
def build_shap_features(actions, rewards):
    """
    Paper: "the sliding window of the past 30 days' actions and rewards
            is used as the feature vector"
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
# CHANGED: agent.model is now a Keras model, so predict_reward must call
# .predict() with the Keras API.  Everything else (KernelExplainer, nsamples)
# is kept identical so this ablation isolates only the DDQN change.
#
# NOTE: In Contribution 4 this will be swapped for shap.DeepExplainer which
# will be ~30x faster and can explain all 253 test steps instead of 100.
def explain_with_shap(agent, X, y):
    def predict_reward(x):
        # Keras .predict() expects numpy arrays and returns numpy arrays.
        # suppress Keras progress bars with verbose=0
        q = agent.model.predict(np.atleast_2d(x), verbose=0)
        return q.max(axis=1)   # max Q-value as reward proxy (unchanged logic)

    bg_idx     = np.random.choice(len(X), min(SHAP_BG, len(X)), replace=False)
    background = X[bg_idx]
    explainer  = shap.KernelExplainer(predict_reward, background)
    shap_vals  = explainer.shap_values(X[:100], nsamples=100, silent=True)
    return shap_vals, X[:100], y[:100]


# ── Aggregate to per-day SHAP ─────────────────────────────────────────────────
# UNCHANGED
def per_day_shap(shap_vals, actions, start_idx=WINDOW_SIZE):
    n_steps     = shap_vals.shape[0]
    day_shap    = np.zeros((n_steps, WINDOW_SIZE))
    day_actions = np.zeros((n_steps, WINDOW_SIZE))

    for t in range(n_steps):
        for d in range(WINDOW_SIZE):
            day_shap[t, d]    = shap_vals[t, d*2] + shap_vals[t, d*2 + 1]
            day_actions[t, d] = actions[start_idx + t - WINDOW_SIZE + d]

    return day_shap, day_actions


def day_names():
    return [f"day_t-{WINDOW_SIZE - d}" for d in range(WINDOW_SIZE)]


# ── Full pipeline ─────────────────────────────────────────────────────────────
# UNCHANGED — just prints "DDQN" in the header for clarity
def run_pipeline(train_data, test_data, tickers_to_run=None):
    if tickers_to_run is None:
        tickers_to_run = list(train_data.keys())

    results = {}
    for ticker in tickers_to_run:
        if ticker not in train_data or ticker not in test_data:
            print(f"[skip] {ticker}"); continue

        print(f"\n{'='*55}\nTicker: {ticker}\n{'='*55}")

        print("Training DDQN (Keras + Adam)...")
        agent, ep_rews = train_agent(ticker, train_data[ticker])
        print(f"  Final ep reward: {ep_rews[-1]:.4f}")

        print("Collecting test experience...")
        actions, rewards = collect_test_experience(test_data[ticker], agent)
        print(f"  Steps={len(actions)}  Buy%={100*actions.mean():.1f}%  "
              f"Total reward={rewards.sum():.4f}")

        X, y = build_shap_features(actions, rewards)
        print(f"  SHAP feature matrix: {X.shape}")

        shap_vals, test_X, test_y = explain_with_shap(agent, X, y)
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


# ── Diagnostic (kept for debugging, updated to DDQNAgent) ────────────────────
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
    results = run_pipeline(train_data, test_data, tickers_to_run=available)

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