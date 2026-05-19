"""
trial/rl/DQN.py – SB3 DQN training for ultrasound probe navigation.

Trains a DQN agent (Stable-Baselines3) using ProbeNavEnv with a CNN policy.
The raw (H, W) float32 observation is downsampled to (84, 84, 1) uint8 before
being fed to the NatureCNN network.

Action space  : Discrete(6) — ±x lateral, ±z elevational, ±θ rotation
Reward        : NCC with target image (frame 566), ∈ [0, 1]
Episode ends  : NCC ≥ 0.95 (success) or step ≥ max_steps (truncation)

Run from the RecON repository root:
    python trial/rl/DQN.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# ── path setup ────────────────────────────────────────────────────────────────
_here    = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(os.path.dirname(_here))
_env_dir = os.path.join(os.path.dirname(_here), "env")
for _p in (_root, _env_dir, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")   # headless backend before any pyplot import

from gym_env import ProbeNavEnv
from rl_env import (
    ProbeEnv,
    probe_surface, volume_up,
    scale_h, scale_w, H_img, W_img,
    example_scan,
)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


# ─────────────────────────────────────────────────────────────────────────────
# Observation wrapper: (H, W) float32 → (size, size, 1) uint8
# ─────────────────────────────────────────────────────────────────────────────

OBS_SIZE = 84   # spatial resolution fed to NatureCNN


class ObsPreprocessWrapper(gym.ObservationWrapper):
    """Downsample greyscale (H, W) float32 obs to (size, size, 1) uint8.

    SB3's CnnPolicy / NatureCNN expects a 3-D uint8 or float32 image space.
    Bilinear downsampling avoids aliasing artefacts from the fan-shaped US image.
    """

    def __init__(self, env: gym.Env, size: int = OBS_SIZE):
        super().__init__(env)
        self._size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 1), dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs : (H, W) float32 ∈ [0, 1]
        t   = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        t   = F.interpolate(t, size=(self._size, self._size),
                            mode="bilinear", align_corners=False)
        arr = t.squeeze().numpy()                               # (size, size)
        return (arr * 255).clip(0, 255).astype(np.uint8)[..., None]  # (size, size, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

TARGET_IDX  = 566    # last acquired frame — navigation goal
MAX_STEPS   = 200
SUCCESS_NCC = 0.95


def make_env(obs_size: int = OBS_SIZE) -> gym.Env:
    """Return a Monitor-wrapped ProbeNavEnv ready for SB3."""
    base = ProbeEnv(
        probe_surface, volume_up, scale_h, scale_w, H_img, W_img, plotter=None,
    )
    target_img = example_scan[TARGET_IDX].astype(np.float32) / 255.0
    env = ProbeNavEnv(
        base, target_img=target_img,
        max_steps=MAX_STEPS, success_threshold=SUCCESS_NCC,
    )
    env = ObsPreprocessWrapper(env, size=obs_size)
    env = Monitor(env)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    out_dir  = os.path.join(_here, "dqn_output")
    log_dir  = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    for d in (log_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    # ── 1. Environments ───────────────────────────────────────────────────────
    print("Building environments …")
    train_env = make_env()
    eval_env  = make_env()

    _tgt = example_scan[TARGET_IDX].astype(np.float32) / 255.0
    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Target image      : shape={_tgt.shape}  "
          f"range=[{_tgt.min():.3f}, {_tgt.max():.3f}]")

    # ── 2. DQN model ──────────────────────────────────────────────────────────
    # CnnPolicy uses NatureCNN (3 conv layers) to encode the (84, 84, 1) obs.
    # buffer_size is kept at 50k (vs. default 1M) to limit memory footprint.
    model = DQN(
        policy                  = "CnnPolicy",
        env                     = train_env,
        learning_rate           = 1e-4,
        buffer_size             = 50_000,
        learning_starts         = 1_000,
        batch_size              = 32,
        tau                     = 1.0,           # hard target-network update
        gamma                   = 0.95,          # matches q_learning.py discount
        train_freq              = 4,
        target_update_interval  = 1_000,
        exploration_fraction    = 0.2,           # ε decays over first 20% of steps
        exploration_initial_eps = 1.0,
        exploration_final_eps   = 0.05,
        optimize_memory_usage   = True,          # store obs as uint8 in buffer
        tensorboard_log         = log_dir,
        verbose                 = 1,
    )

    # ── 3. Callbacks ──────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = ckpt_dir,
        name_prefix = "dqn_probe",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = out_dir,
        log_path             = log_dir,
        eval_freq            = 10_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    TOTAL_STEPS = 200_000
    print(f"\nTraining DQN for {TOTAL_STEPS:,} steps …")
    print(f"  Checkpoints → {ckpt_dir}")
    print(f"  TensorBoard → tensorboard --logdir {log_dir}\n")
    model.learn(
        total_timesteps = TOTAL_STEPS,
        callback        = [checkpoint_cb, eval_cb],
        progress_bar    = True,
    )

    # ── 5. Save final model ───────────────────────────────────────────────────
    final_path = os.path.join(out_dir, "dqn_probe_final")
    model.save(final_path)
    print(f"\nSaved final model → {final_path}.zip")

    # ── 6. Evaluate greedy policy ─────────────────────────────────────────────
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True,
    )
    print(f"Final evaluation  mean_reward={mean_r:.4f}  std={std_r:.4f}")

    train_env.close()
    eval_env.close()
