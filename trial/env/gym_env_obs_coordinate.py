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
Reward      : Normalised Cross-Correlation (NCC) with target image, ∈ [0, 1].

Run from the RecON repository root:
    python trial/env/gym_env.py
"""

import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_env import (
    ProbeEnv,
    probe_surface, volume_up,
    scale_h, scale_w, H_img, W_img,
)


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
    success_threshold : NCC threshold that triggers episode termination (success)
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

    def _ncc(self, img: np.ndarray) -> float:
        """Normalised Cross-Correlation mapped from [−1, 1] to [0, 1]."""
        a = img.ravel().astype(np.float64)
        b = self._target.ravel().astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float((np.dot(a, b) / (na * nb) + 1.0) * 0.5)

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

        obs_t       = self._env.reset(s=s0, t=t0, angle=angle0)
        self._obs   = obs_t.numpy().astype(np.float32)
        self._steps = 0
        info = {
            's': s0, 't': t0, 'angle': angle0,
            'ncc': self._ncc(self._obs),
        }
        return self._obs.copy(), info

    def step(self, action: int):
        """Apply one of 6 discrete actions.

        Returns
        -------
        obs        : np.ndarray (H, W) float32
        reward     : float  – NCC with target image, ∈ [0, 1]
        terminated : bool   – NCC ≥ success_threshold
        truncated  : bool   – step count ≥ max_steps
        info       : dict   – {'s', 't', 'angle', 'ncc'}
        """
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected integer in [0, 5].")

        dx, dz, dtheta = self._ACTIONS[action]
        obs_t, state   = self._env.step(dx, dz, dtheta)
        self._obs       = obs_t.numpy().astype(np.float32)
        self._steps    += 1

        ncc        = self._ncc(self._obs)
        reward     = float(ncc)
        terminated = bool(ncc >= self._success_thresh)
        truncated  = bool(self._steps >= self._max_steps)
        return self._obs.copy(), reward, terminated, truncated, {**state, 'ncc': ncc}

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
    target_obs = _base_env.reset(s=0.5, t=1.0, angle=0.0).numpy().astype(np.float32)
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

    print("\n" + "=" * 60)
    print("Spaces")
    print("=" * 60)
    print(f"  observation_space : {env.observation_space}")
    print(f"  action_space      : {env.action_space}")

    # ── 4. reset() ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("reset()")
    print("=" * 60)
    obs, info = env.reset()
    print(f"  obs   shape={obs.shape}  dtype={obs.dtype}"
          f"  range=[{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  info: s={info['s']:.3f}  t={info['t']:.3f}"
          f"  angle={np.degrees(info['angle']):.1f}°  NCC={info['ncc']:.4f}")

    # ── 5. step(): each of the 6 actions once ────────────────────────────────
    print("\n" + "=" * 60)
    print("step() – each action once from the same initial pose")
    print("=" * 60)
    action_labels = ["+x", "-x", "+z", "-z", "+θ", "-θ"]
    for a in range(6):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(a)
        print(f"  action {a} ({action_labels[a]:2s}): "
              f"reward={reward:.4f}  NCC={info['ncc']:.4f}"
              f"  s={info['s']:.4f}  t={info['t']:.4f}"
              f"  terminated={terminated}  truncated={truncated}")

    # ── 6. Random-policy rollout ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Random rollout (up to 50 steps)")
    print("=" * 60)
    obs, info = env.reset()
    rng = np.random.default_rng(42)
    total_reward = 0.0
    nccs = [info['ncc']]
    for i in range(50):
        action = int(rng.integers(0, 6))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        nccs.append(info['ncc'])
        if i < 5 or terminated or truncated:
            print(f"  step {i+1:3d}  action={action}({action_labels[action]:2s})"
                  f"  reward={reward:.4f}  NCC={info['ncc']:.4f}"
                  f"  t={info['t']:.3f}"
                  f"{'  TERMINATED' if terminated else ''}"
                  f"{'  TRUNCATED'  if truncated  else ''}")
        if terminated or truncated:
            break
    print(f"  Cumulative reward: {total_reward:.4f}  Steps: {env._steps}")

    # ── 7. Visualise initial / target frames side-by-side ────────────────────
    obs_init, _ = env.reset()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("ProbeNavEnv  –  initial vs target image", fontsize=11)
    axes[0].imshow(obs_init, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(
        f"Initial pose  (t=0.0)\nNCC = {env._ncc(obs_init):.4f}", fontsize=9,
    )
    axes[0].axis('off')
    axes[1].imshow(target_obs, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Target pose  (t=1.0)\nexample_scan[566]", fontsize=9)
    axes[1].axis('off')
    plt.tight_layout()
    out_path = os.path.join(_here, "probe_nav_env_demo.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved demo figure → {out_path}")
    plt.show()
    print("Done.")
