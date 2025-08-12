from __future__ import annotations
from pathlib import Path
import argparse, numpy as np, pandas as pd
try:
    import gymnasium as gym  # noqa
except ImportError:
    import gym  # noqa
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from credit_limit_env import CreditLimitEnv, EnvConfig
from synthetic_data import SynthConfig, generate_all
from data_io import load_from_parquet, load_from_hive

def make_env(customers: pd.DataFrame, macro: pd.DataFrame, seed: int = 0, months: int = 60):
    def _thunk():
        cfg = EnvConfig(seed=seed, months=months)
        return CreditLimitEnv(customers, macro, cfg)
    return _thunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", choices=["synthetic","parquet","hive"], default="synthetic")
    ap.add_argument("--parquet-dir", default="data")
    ap.add_argument("--hive-db", default="credit_rl")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--parallel-envs", type=int, default=8)
    ap.add_argument("--device", default="cuda")  # set "cpu" if no GPU
    args = ap.parse_args()

    out = Path("outputs"); (out/"logs").mkdir(parents=True, exist_ok=True); (out/"models").mkdir(parents=True, exist_ok=True)

    # Load data
    if args.data_source=="synthetic":
        customers, macro = generate_all(SynthConfig(num_customers=1000, months=60, seed=7))
    elif args.data_source=="parquet":
        customers, macro = load_from_parquet(args.parquet_dir)
    else:
        customers, macro = load_from_hive(args.hive_db)

    # Vec envs
    env_fns = [make_env(customers, macro, seed=s, months=60) for s in range(args.parallel_envs)]
    vec = DummyVecEnv(env_fns)
    vec = VecMonitor(vec, filename=str(out/"logs"/"monitor.csv"))
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", vec, device=args.device,
        n_steps=2048, batch_size=2048, learning_rate=3e-4,
        gamma=0.995, gae_lambda=0.95, ent_coef=0.02, clip_range=0.2,
        policy_kwargs=dict(net_arch=[dict(pi=[256,256], vf=[256,256])]),
        tensorboard_log=str(out/"logs"/"tb"), verbose=1, seed=0,
    )
    model.set_logger(configure(str(out/"logs"), ["stdout","csv","tensorboard"]))
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model.save(out/"models"/"ppo_credit_limit.zip")
    np.savez(out/"models"/"vec_stats.npz", mean=vec.obs_rms.mean, var=vec.obs_rms.var, clip_obs=vec.clip_obs)
    print("Saved:", out/"models"/"ppo_credit_limit.zip")

if __name__ == "__main__":
    main()
