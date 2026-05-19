"""
trial/rl/q_learning.py – Q-learning training for ProbeNavEnv.

Follows the Gymnasium tutorial approach:
https://gymnasium.farama.org/introduction/train_agent/

The observation space is continuous (H×W image), so the probe's continuous
(s, t, angle) coordinates from info are discretised into a hashable Q-table key.

Run from the RecON repository root:
    python trial/rl/q_learning.py
"""

import os
import sys
import pickle
from collections import defaultdict
from typing import Tuple

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
_here    = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(os.path.dirname(_here))
_env_dir = os.path.join(os.path.dirname(_here), "env")
for _p in (_root, _env_dir, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gym_env import ProbeNavEnv          # Gymnasium wrapper (trial/rl/gym_env.py)
from rl_env import (                     # low-level sampler  (trial/env/rl_env.py)
    ProbeEnv,
    probe_surface, volume_up,
    scale_h, scale_w, H_img, W_img,
    example_scan,
)

# ── state discretisation ──────────────────────────────────────────────────────
_N_BINS      = 20
_ANGLE_RANGE = np.radians(45.0)          # clip ±45° before binning


def obs_key(info: dict) -> Tuple[int, int, int]:
    """Map continuous probe state (s, t, angle) → discrete (int, int, int) key."""
    s_bin  = int(np.clip(info["s"] * _N_BINS, 0, _N_BINS - 1))
    t_bin  = int(np.clip(info["t"] * _N_BINS, 0, _N_BINS - 1))
    a_norm = (np.clip(info["angle"], -_ANGLE_RANGE, _ANGLE_RANGE) + _ANGLE_RANGE) / (
        2 * _ANGLE_RANGE
    )
    a_bin  = int(np.clip(a_norm * _N_BINS, 0, _N_BINS - 1))
    return (s_bin, t_bin, a_bin)


# ── Q-learning agent ──────────────────────────────────────────────────────────

class ProbeNavAgent:
    """Tabular Q-learning agent for ProbeNavEnv.

    Follows the BlackjackAgent structure from the Gymnasium training tutorial.
    State keys are discrete (s_bin, t_bin, a_bin) triples produced by obs_key().
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env             = env
        self.q_values        = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr              = learning_rate
        self.discount_factor = discount_factor
        self.epsilon         = initial_epsilon
        self.epsilon_decay   = epsilon_decay
        self.final_epsilon   = final_epsilon
        self.training_error  = []

    def get_action(self, obs: Tuple[int, int, int]) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: Tuple[int, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Tuple[int, int, int],
    ):
        """Bellman update for the visited (state, action) pair."""
        future_q  = (not terminated) * np.max(self.q_values[next_obs])
        target    = reward + self.discount_factor * future_q
        td        = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * td
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# ── visualisation ─────────────────────────────────────────────────────────────

def _moving_avg(arr, window, mode="valid"):
    return np.convolve(np.array(arr).flatten(), np.ones(window), mode=mode) / window


def plot_training(env, agent, rolling_length: int = 500, save_path: str = None):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    fig.suptitle("ProbeNavEnv – Q-learning training curves", fontsize=11)

    axs[0].set_title("Episode rewards")
    axs[0].plot(_moving_avg(env.return_queue, rolling_length))
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Average NCC reward")

    axs[1].set_title("Episode lengths")
    axs[1].plot(_moving_avg(env.length_queue, rolling_length))
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Average steps")

    axs[2].set_title("Training error (TD)")
    axs[2].plot(_moving_avg(agent.training_error, rolling_length, mode="same"))
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("TD error")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved training curves → {save_path}")
    plt.close(fig)


# ── navigation visualisation ──────────────────────────────────────────────────

_ACTION_LABELS = ["+x", "-x", "+z", "-z", "+θ", "-θ"]


def visualize_navigation(
    agent: ProbeNavAgent,
    env: gym.Env,
    save_path: str = None,
    episode: int = 0,
):
    """Run one greedy episode and save a navigation summary figure.

    Layout
    ------
    Top row  : 5 evenly-spaced trajectory snapshots  |  target image
    Bottom   : NCC curve  |  (s,t) trajectory (NCC-coloured)  |  angle curve
    """
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    obs, info = env.reset()
    state = obs_key(info)
    obs_history    = [obs]
    info_history   = [info]
    action_history = []

    done = False
    while not done:
        action = agent.get_action(state)
        obs, _r, terminated, truncated, info = env.step(action)
        state = obs_key(info)
        obs_history.append(obs)
        info_history.append(info)
        action_history.append(action)
        done = terminated or truncated

    agent.epsilon = old_eps

    nccs   = [d["ncc"]               for d in info_history]
    ss     = [d["s"]                 for d in info_history]
    ts     = [d["t"]                 for d in info_history]
    angles = [np.degrees(d["angle"]) for d in info_history]
    n_steps = len(action_history)
    steps   = list(range(len(nccs)))

    # 5 evenly-spaced trajectory snapshots + target image = 6 columns total
    N_COLS  = 6
    n_snap  = min(5, len(obs_history))
    snap_idxs = np.linspace(0, len(obs_history) - 1, n_snap, dtype=int)

    status = "Success" if nccs[-1] >= 0.95 else f"NCC={nccs[-1]:.3f}"
    fig = plt.figure(figsize=(N_COLS * 2.5, 7))
    fig.suptitle(
        f"Q-learning Navigation  –  Episode {episode}  "
        f"({status},  {n_steps} steps)",
        fontsize=11,
    )
    gs = fig.add_gridspec(2, N_COLS, hspace=0.5, wspace=0.35)

    # ── Row 0: trajectory snapshots (cols 0-4) + target (col 5) ───────────────
    for col, fi in enumerate(snap_idxs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(obs_history[fi], cmap="gray", vmin=0, vmax=1)
        act_str = (
            f"\na={_ACTION_LABELS[action_history[fi - 1]]}" if fi > 0 else ""
        )
        ax.set_title(f"step {fi}  NCC={nccs[fi]:.3f}{act_str}", fontsize=7)
        ax.axis("off")

    ax_tgt = fig.add_subplot(gs[0, N_COLS - 1])
    ax_tgt.imshow(env._target, cmap="gray", vmin=0, vmax=1)
    ax_tgt.set_title("Target image", fontsize=7)
    ax_tgt.axis("off")

    # ── Row 1 left (cols 0-1): NCC curve ──────────────────────────────────────
    ax_ncc = fig.add_subplot(gs[1, 0:2])
    ax_ncc.plot(steps, nccs, color="steelblue", lw=1.5)
    ax_ncc.axhline(0.95, color="red", lw=1, ls="--", label="threshold 0.95")
    ax_ncc.set_xlabel("Step")
    ax_ncc.set_ylabel("NCC")
    ax_ncc.set_title("NCC over steps")
    ax_ncc.legend(fontsize=7)
    ax_ncc.set_ylim(0, 1.05)

    # ── Row 1 middle (cols 2-3): (s, t) trajectory ────────────────────────────
    ax_st = fig.add_subplot(gs[1, 2:4])
    sc = ax_st.scatter(ss, ts, c=nccs, cmap="RdYlGn", vmin=0, vmax=1, s=15, zorder=3)
    ax_st.plot(ss, ts, color="gray", lw=0.8, zorder=2, alpha=0.6)
    ax_st.scatter([ss[0]],  [ts[0]],  marker="o", s=80,  c="blue", zorder=4, label="start")
    ax_st.scatter([ss[-1]], [ts[-1]], marker="*", s=120, c="red",  zorder=4, label="end")
    plt.colorbar(sc, ax=ax_st, fraction=0.046, pad=0.04, label="NCC")
    ax_st.set_xlabel("s (lateral)")
    ax_st.set_ylabel("t (along-scan)")
    ax_st.set_xlim(-0.05, 1.05)
    ax_st.set_ylim(-0.05, 1.05)
    ax_st.set_title("Probe trajectory  (s, t)")
    ax_st.legend(fontsize=7)

    # ── Row 1 right (cols 4-5): rotation angle ────────────────────────────────
    ax_ang = fig.add_subplot(gs[1, 4:6])
    ax_ang.plot(steps, angles, color="darkorange", lw=1.5)
    ax_ang.set_xlabel("Step")
    ax_ang.set_ylabel("Angle (°)")
    ax_ang.set_title("Rotation angle over steps")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved navigation visualisation → {save_path}")
    plt.close(fig)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(agent: ProbeNavAgent, env: gym.Env, n_episodes: int = 200):
    """Run the greedy policy and report mean NCC and steps."""
    old_eps     = agent.epsilon
    agent.epsilon = 0.0
    rewards, lengths = [], []

    for _ in range(n_episodes):
        _, info = env.reset()
        state   = obs_key(info)
        ep_r    = 0.0
        steps   = 0
        done    = False
        while not done:
            action = agent.get_action(state)
            _, r, terminated, truncated, info = env.step(action)
            state  = obs_key(info)
            ep_r  += r
            steps += 1
            done   = terminated or truncated
        rewards.append(ep_r)
        lengths.append(steps)

    agent.epsilon = old_eps
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Mean cumulative NCC reward : {np.mean(rewards):.4f}")
    print(f"  Mean episode length        : {np.mean(lengths):.1f} steps")
    print(f"  Q-table states visited     : {len(agent.q_values)}")


# ── main ──────────────────────────────────────────────────────────────────────

def test_saved_agent(
    qt_path: str = None,
    n_episodes: int = 20,
    n_vis: int = 1,
    vis_dir: str = None,
):
    """Load q_table.pkl, visualise n_vis episodes, then evaluate.

    Parameters
    ----------
    qt_path    : path to saved Q-table pickle (default: q_table.pkl next to this file)
    n_episodes : total evaluation episodes (includes visualised ones)
    n_vis      : how many episodes to visualise as PNG (0 = skip)
    vis_dir    : directory for output PNGs (default: same directory as this file)
    """
    from collections import defaultdict
    if qt_path is None:
        qt_path = os.path.join(_here, "q_table.pkl")
    if vis_dir is None:
        vis_dir = _here

    base_env = ProbeEnv(
        probe_surface, volume_up, scale_h, scale_w, H_img, W_img, plotter=None,
    )
    target_img = example_scan[566].astype(np.float32) / 255.0
    raw_env = ProbeNavEnv(base_env, target_img=target_img, max_steps=200, success_threshold=0.95)

    with open(qt_path, "rb") as f:
        q_dict = pickle.load(f)
    print(f"Loaded Q-table: {len(q_dict)} states  ←  {qt_path}")

    agent = ProbeNavAgent(
        env=raw_env,
        learning_rate=0.0,          # no updates during test
        initial_epsilon=0.0,        # fully greedy
        epsilon_decay=0.0,
        final_epsilon=0.0,
    )
    agent.q_values = defaultdict(lambda: np.zeros(raw_env.action_space.n), q_dict)

    n_vis = min(n_vis, n_episodes)
    for ep in range(n_vis):
        vis_path = os.path.join(vis_dir, f"navigation_ep{ep:03d}.png")
        visualize_navigation(agent, raw_env, save_path=vis_path, episode=ep)

    remaining = n_episodes - n_vis
    if remaining > 0:
        evaluate(agent, raw_env, n_episodes=remaining)


if __name__ == "__main__":
    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--test", action="store_true", help="evaluate saved q_table.pkl instead of training")
    # ap.add_argument("--n_eval", type=int, default=20, help="number of evaluation episodes")
    # args = ap.parse_args()
    #
    # if args.test:
    #     test_saved_agent(n_episodes=args.n_eval)
    #     import sys; sys.exit(0)

    # test
    # test_saved_agent(n_episodes=1)

    # train
    # 1. Build the low-level probe sampler (headless)
    base_env = ProbeEnv(
        probe_surface, volume_up, scale_h, scale_w, H_img, W_img, plotter=None,
    )

    # 2. Target image: last acquired frame (index 566)
    target_img = example_scan[566].astype(np.float32) / 255.0
    print(f"Target image  shape={target_img.shape}  "
          f"range=[{target_img.min():.3f}, {target_img.max():.3f}]")

    # 3. Hyperparameters
    learning_rate   = 0.1
    # n_episodes      = 100_000
    n_episodes      = 1000
    start_epsilon   = 1.0
    epsilon_decay   = start_epsilon / (n_episodes / 2)
    final_epsilon   = 0.1
    discount_factor = 0.95
    max_steps       = 50

    # 4. Gymnasium env with episode statistics wrapper
    _raw_env = ProbeNavEnv(base_env, target_img=target_img,
                           max_steps=max_steps, success_threshold=0.95)
    env = gym.wrappers.RecordEpisodeStatistics(_raw_env, buffer_length=n_episodes)

    # 5. Agent
    agent = ProbeNavAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    # 6. Training loop  (Gymnasium tutorial style)
    for episode in tqdm(range(n_episodes), desc="Training"):
        _, info = env.reset()
        state   = obs_key(info)
        done    = False

        while not done:
            action              = agent.get_action(state)
            _, reward, terminated, truncated, info = env.step(action)
            next_state          = obs_key(info)
            agent.update(state, action, reward, terminated, next_state)
            done                = terminated or truncated
            state               = next_state

        agent.decay_epsilon()

    # 7. Save Q-table
    qt_path = os.path.join(_here, "q_table.pkl")
    with open(qt_path, "wb") as f:
        pickle.dump(dict(agent.q_values), f)
    print(f"\nSaved Q-table ({len(agent.q_values)} states) → {qt_path}")

    # 8. Plot training curves
    plot_training(
        env, agent,
        rolling_length=500,
        save_path=os.path.join(_here, "q_learning_curves.png"),
    )

    # # 9. Evaluate the greedy policy
    # evaluate(agent, _raw_env, n_episodes=10)
