from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# Gymnasium preferred (not directly used here but good to keep consistent)
try:
    import gymnasium as gym  # noqa: F401
except ImportError:
    import gym  # noqa: F401

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from credit_limit_env import CreditLimitEnv, EnvConfig
from synthetic_data import SynthConfig, generate_all


def make_env(customers: pd.DataFrame, macro: pd.DataFrame, seed: int = 0):
    def _thunk():
        # Horizon aligned to evaluation (60)
        cfg = EnvConfig(seed=seed, months=60)
        env = CreditLimitEnv(customers, macro, cfg)
        return env
    return _thunk


def main():
    np.random.seed(0)

    out_dir = Path("outputs")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # Data (with an optional macro shock mid-way to add variety)
    synth_cfg = SynthConfig(num_customers=1000, months=60, shock_enabled=True, shock_month=24)
    customers, macro = generate_all(synth_cfg)

    # Parallel vectorized envs
    env_fns = [make_env(customers, macro, seed=s) for s in range(8)]
    vec = DummyVecEnv(env_fns)
    vec = VecMonitor(vec, filename=str(out_dir / "logs" / "monitor.csv"))
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # PPO model on GPU with a slightly larger net/batches for stability
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        device="cuda",                   # use your NVIDIA GPU
        n_steps=2048,
        batch_size=2048,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        tensorboard_log=str(out_dir / "logs" / "tb"),
        verbose=1,
        seed=0,
    )

    # Custom logger (stdout + csv + TB)
    new_logger = configure(str(out_dir / "logs"), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Train (~1M timesteps; adjust down if you want it faster)
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model
    model_path = out_dir / "models" / "ppo_credit_limit.zip"
    model.save(model_path)

    # Save VecNormalize stats (so we can normalize obs at evaluation)
    stats_path = out_dir / "models" / "vec_stats.npz"
    np.savez(
        stats_path,
        mean=vec.obs_rms.mean,
        var=vec.obs_rms.var,
        clip_obs=vec.clip_obs,
    )

    print(f"Saved model to: {model_path}")
    print(f"Saved normalization stats to: {stats_path}")


if __name__ == "__main__":
    main()
