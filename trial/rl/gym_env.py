"""
trial/env/gym_env.py  –  Gymnasium environment for US probe navigation.

Goal: navigate the probe to the target pose corresponding to example_scan[566]
(the last acquired ultrasound frame).  The observation at that pose is obtained
by sampling the reconstructed 3-D volume, so it is fully differentiable from
the pose parameterisation.

Discrete action space (6 actions):
    0  +2 mm along probe x-axis (lateral,    +x)
    1  −2 mm along probe x-axis (lateral,    −x)
    2  +2 mm along probe z-axis (elevational, +z)
    3  −2 mm along probe z-axis (elevational, −z)
    4  +2° around probe y-axis  (outward normal, CCW)
    5  −2° around probe y-axis  (outward normal, CW)

Observation : (H, W) float32 ∈ [0, 1]  simulated US slice at current pose.
Reward      : Structural Similarity Index (SSIM) with target image, ∈ [0, 1].

Run from the RecON repository root:
    python trial/env/gym_env.py
"""

import sys
import os
from typing import Tuple

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
_env_dir = os.path.join(os.path.dirname(_here), 'env')
for _p in (_root, _env_dir, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from skimage.metrics import structural_similarity as ssim_metric

from rl_env import (
    ProbeEnv,
    probe_surface, volume_up,
    scale_h, scale_w, H_img, W_img,
    example_scan,
)

from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium environment
# ─────────────────────────────────────────────────────────────────────────────

class ProbeNavEnv(gym.Env):
    """Gymnasium environment: navigate the US probe to a target pose.

    The probe pose is parameterised by (s, t, angle):
      s     ∈ [0, 1]  lateral position on the contact surface
      t     ∈ [0, 1]  along-scan position  (0 = first frame, 1 = last frame)
      angle ∈ ℝ       rotation around the outward surface normal (radians)

    Observation space : Box(0, 1, shape=(H, W), dtype=float32)
    Action space      : Discrete(6)

    Parameters
    ----------
    probe_env         : pre-built ProbeEnv (low-level slice sampler)
    target_img        : (H, W) float32 target US image
    max_steps         : episode truncation limit
    success_threshold : SSIM threshold that triggers episode termination (success)
    render_mode       : None | "rgb_array"
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    # Each entry: (dx_mm, dz_mm, dtheta_rad)
    _ACTIONS = [
        ( 2.0,   0.0,   0.0             ),  # 0: +x  lateral +2 mm
        (-2.0,   0.0,   0.0             ),  # 1: -x  lateral -2 mm
        ( 0.0,   2.0,   0.0             ),  # 2: +z  elevational +2 mm
        ( 0.0,  -2.0,   0.0             ),  # 3: -z  elevational -2 mm
        ( 0.0,   0.0,   np.radians(2.0) ),  # 4: +θ  CCW +2°
        ( 0.0,   0.0,  -np.radians(2.0) ),  # 5: -θ  CW  -2°
    ]

    def __init__(
        self,
        probe_env: ProbeEnv,
        target_img: np.ndarray,
        max_steps: int = 200,
        success_threshold: float = 0.95,
        render_mode=None,
    ):
        super().__init__()
        self._env            = probe_env
        self._target         = target_img.astype(np.float32)
        self._max_steps      = max_steps
        self._success_thresh = success_threshold
        self.render_mode     = render_mode

        H, W = probe_env._H, probe_env._W
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(H, W), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

        self._steps = 0
        self._obs   = np.zeros((H, W), dtype=np.float32)

    # ── reward ────────────────────────────────────────────────────────────────

    def _ssim(self, img: np.ndarray) -> float:
        """Structural Similarity Index (SSIM), ∈ [0, 1] for uint8-scaled inputs."""
        return float(ssim_metric(
            img.astype(np.float64),
            self._target.astype(np.float64),
            data_range=1.0,
        ))

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        """Reset probe to initial pose and return (obs, info).

        options keys (all optional):
            's'     float ∈ [0,1]  lateral position on surface  (default 0.5)
            't'     float ∈ [0,1]  along-scan position          (default 0.0)
            'angle' float          rotation angle in radians     (default 0.0)
        """
        super().reset(seed=seed)
        s0     = 0.5
        t0     = 0.0
        angle0 = 0.0
        if options:
            s0     = float(options.get('s',     s0))
            t0     = float(options.get('t',     t0))
            angle0 = float(options.get('angle', angle0))

        obs_t           = self._env.reset(s=s0, t=t0, angle=angle0)
        self._obs       = obs_t.numpy().astype(np.float32)
        self._steps     = 0
        self._prev_ssim = self._ssim(self._obs)
        info = {
            's': s0, 't': t0, 'angle': angle0,
            'ssim': self._prev_ssim,
        }
        return self._obs.copy(), info

    def step(self, action: int):
        """Apply one of 6 discrete actions.

        Returns
        -------
        obs        : np.ndarray (H, W) float32
        reward     : float  – SSIM with target image, ∈ [0, 1]
        terminated : bool   – SSIM ≥ success_threshold
        truncated  : bool   – step count ≥ max_steps
        info       : dict   – {'s', 't', 'angle', 'ssim'}
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected integer in [0, 5].")

        dx, dz, dtheta  = self._ACTIONS[action]
        obs_t, state    = self._env.step(dx, dz, dtheta)
        self._obs        = obs_t.numpy().astype(np.float32)
        self._steps     += 1

        sim              = self._ssim(self._obs)
        reward           = float(sim - self._prev_ssim)
        self._prev_ssim  = sim
        terminated       = bool(sim >= self._success_thresh)
        truncated        = bool(self._steps >= self._max_steps)
        return self._obs.copy(), reward, terminated, truncated, {**state, 'ssim': sim}

    def render(self):
        """Return current observation as (H, W, 3) uint8 array (render_mode='rgb_array')."""
        if self.render_mode == "rgb_array":
            u8 = (self._obs * 255).clip(0, 255).astype(np.uint8)
            return np.stack([u8] * 3, axis=-1)
        return None

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ── 1. Build the headless base environment ────────────────────────────────
    _base_env = ProbeEnv(
        probe_surface, volume_up, scale_h, scale_w, H_img, W_img,
        plotter=None,
    )

    # ── 2. Target image: simulated slice at t=1.0 (→ example_scan[566]) ──────
    # t=1.0 maps to the last valid frame in the contact-surface parameterisation,
    # which corresponds to example_scan[566].
    target_obs = example_scan[566].astype(np.float32) / 255.0
    print(f"Target image  shape={target_obs.shape}  "
          f"range=[{target_obs.min():.3f}, {target_obs.max():.3f}]")

    # ── 3. Instantiate ProbeNavEnv ────────────────────────────────────────────
    env = ProbeNavEnv(
        _base_env,
        target_img=target_obs,
        max_steps=200,
        success_threshold=0.95,
        render_mode="rgb_array",
    )

    action_labels = ["+x", "-x", "+z", "-z", "+θ", "-θ"]


    class Agent:
        def __init__(
                self,
                env: gym.Env,
                learning_rate: float,
                initial_epsilon: float,
                epsilon_decay: float,
                final_epsilon: float,
                discount_factor: float = 0.95,
        ):
            """Initialize a Q-Learning agent.

            Args:
                env: The training environment
                learning_rate: How quickly to update Q-values (0-1)
                initial_epsilon: Starting exploration rate (usually 1.0)
                epsilon_decay: How much to reduce epsilon each episode
                final_epsilon: Minimum exploration rate (usually 0.1)
                discount_factor: How much to value future rewards (0-1)
            """
            self.env = env

            # Q-table: maps (state, action) to expected reward
            # defaultdict automatically creates entries with zeros for new states
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

            self.lr = learning_rate
            self.discount_factor = discount_factor  # How much we care about future rewards

            # Exploration parameters
            self.epsilon = initial_epsilon
            self.epsilon_decay = epsilon_decay
            self.final_epsilon = final_epsilon

            # Track learning progress
            self.training_error = []

        def get_action(self, obs: Tuple[int, int, int]) -> int:
            """Choose an action using epsilon-greedy strategy.

            Returns:
                action: 0 (stand) or 1 (hit)
            """
            # With probability epsilon: explore (random action)
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()

            # With probability (1-epsilon): exploit (best known action)
            else:
                return int(np.argmax(self.q_values[obs]))

        def update(
                self,
                obs: Tuple[int, int, int],
                action: int,
                reward: float,
                terminated: bool,
                next_obs: Tuple[int, int, int],
        ):
            """Update Q-value based on experience.

            This is the heart of Q-learning: learn from (state, action, reward, next_state)
            """
            # What's the best we could do from the next state?
            # (Zero if episode terminated - no future rewards possible)
            future_q_value = (not terminated) * np.max(self.q_values[next_obs])

            # What should the Q-value be? (Bellman equation)
            target = reward + self.discount_factor * future_q_value

            # How wrong was our current estimate?
            temporal_difference = target - self.q_values[obs][action]

            # Update our estimate in the direction of the error
            # Learning rate controls how big steps we take
            self.q_values[obs][action] = (
                    self.q_values[obs][action] + self.lr * temporal_difference
            )

            # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)

        def decay_epsilon(self):
            """Reduce exploration rate after each episode."""
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    _N_BINS = 20
    _ANGLE_RANGE = np.radians(45.0)  # clip angle to ±45° before binning

    def _obs_key(info: dict) -> Tuple[int, int, int]:
        """Convert continuous probe state to a hashable discrete Q-table key."""
        s_bin = int(np.clip(info['s'] * _N_BINS, 0, _N_BINS - 1))
        t_bin = int(np.clip(info['t'] * _N_BINS, 0, _N_BINS - 1))
        a_norm = (np.clip(info['angle'], -_ANGLE_RANGE, _ANGLE_RANGE) + _ANGLE_RANGE) / (2 * _ANGLE_RANGE)
        a_bin = int(np.clip(a_norm * _N_BINS, 0, _N_BINS - 1))
        return (s_bin, t_bin, a_bin)

    # Training hyperparameters
    learning_rate = 0.01  # How fast to learn (higher = faster but less stable)
    n_episodes = 100_000  # Number of hands to practice
    start_epsilon = 1.0  # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1  # Always keep some exploration

    # Create environment and agent
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Agent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    from tqdm import tqdm  # Progress bar

    for episode in tqdm(range(n_episodes)):
        _, info = env.reset()
        obs = _obs_key(info)
        done = False

        while not done:
            action = agent.get_action(obs)

            _, reward, terminated, truncated, info = env.step(action)
            next_obs = _obs_key(info)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    # # ── 6. Random-policy rollout ──────────────────────────────────────────────
    # print("\n" + "=" * 60)
    # print("Random rollout (up to 50 steps)")
    # print("=" * 60)
    # obs, info = env.reset()
    # rng = np.random.default_rng(42)
    # total_reward = 0.0
    # nccs = [info['ssim']]
    # for i in range(50):
    #     action = int(rng.integers(0, 6))
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
    #     nccs.append(info['ssim'])
    #     if i < 5 or terminated or truncated:
    #         print(f"  step {i+1:3d}  action={action}({action_labels[action]:2s})"
    #               f"  reward={reward:.4f}  SSIM={info['ssim']:.4f}"
    #               f"  t={info['t']:.3f}"
    #               f"{'  TERMINATED' if terminated else ''}"
    #               f"{'  TRUNCATED'  if truncated  else ''}")
    #     if terminated or truncated:
    #         break
    # print(f"  Cumulative reward: {total_reward:.4f}  Steps: {env._steps}")
    #
    # # ── 7. Visualise initial / target frames side-by-side ────────────────────
    # obs_init, _ = env.reset()
    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # fig.suptitle("ProbeNavEnv  –  initial vs target image", fontsize=11)
    # axes[0].imshow(obs_init, cmap='gray', vmin=0, vmax=1)
    # axes[0].set_title(
    #     f"Initial pose  (t=0.0)\nSSIM = {env._ssim(obs_init):.4f}", fontsize=9,
    # )
    # axes[0].axis('off')
    # axes[1].imshow(target_obs, cmap='gray', vmin=0, vmax=1)
    # axes[1].set_title("Target pose  (t=1.0)\nexample_scan[566]", fontsize=9)
    # axes[1].axis('off')
    # plt.tight_layout()
    # out_path = os.path.join(_here, "probe_nav_env_demo.png")
    # plt.savefig(out_path, dpi=120)
    # print(f"\nSaved demo figure → {out_path}")
    # plt.show()
    # print("Done.")
