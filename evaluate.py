from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from stable_baselines3 import PPO
from credit_limit_env import simulate_accounts, EnvConfig
from synthetic_data import SynthConfig, generate_all
from data_io import load_from_parquet, load_from_hive
from baseline_policy import baseline_rule_factory

def load_vec_stats(path: Path):
    if path.exists():
        d = np.load(path)
        return dict(mean=d["mean"], var=d["var"], clip_obs=float(d["clip_obs"]))
    return None

def rl_policy_fn_factory(model: PPO, vec_stats: dict | None):
    def policy_fn(state: dict) -> int:
        obs = np.array([
            state["limit_norm"], state["avg_spend_norm"], state["repay_rate"],
            state["util"], state["interest_rate"], state["unemployment_rate"],
        ], dtype=np.float32).reshape(1, -1)
        if vec_stats is not None:
            obs = (obs - vec_stats["mean"]) / np.sqrt(vec_stats["var"] + 1e-8)
            obs = np.clip(obs, -vec_stats["clip_obs"], vec_stats["clip_obs"])
        action, _ = model.predict(obs, deterministic=True)
        return int(action[0])
    return policy_fn

def compute_metrics(panel: pd.DataFrame):
    by_m = panel.groupby("month").agg(
        profit_m=("profit","sum"),
        reward_m=("reward","sum"),
        defaults=("default","sum"),
        avg_limit=("limit","mean"),
        avg_defprob=("default_prob","mean"),
    ).reset_index()
    by_m["cum_profit"] = by_m["profit_m"].cumsum()
    by_m["cum_reward"] = by_m["reward_m"].cumsum()
    summary = dict(
        total_profit=float(panel["profit"].sum()),
        total_reward=float(panel["reward"].sum()),
        total_defaults=int(panel["default"].sum()),
        avg_limit=float(panel["limit"].mean()),
        avg_default_prob=float(panel["default_prob"].mean()),
    )
    return by_m, summary

def pct_improve(rl, base): return (100.0*(rl-base)/abs(base)) if base!=0 else (np.inf if rl>0 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/models/ppo_credit_limit.zip")
    ap.add_argument("--data-source", choices=["synthetic","parquet","hive"], default="synthetic")
    ap.add_argument("--parquet-dir", default="data")
    ap.add_argument("--hive-db", default="credit_rl")
    ap.add_argument("--customers", type=int, default=1000)
    ap.add_argument("--months", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shock", action="store_true")
    args = ap.parse_args()

    outdir = Path("outputs/eval"); outdir.mkdir(parents=True, exist_ok=True)

    if args.data_source=="synthetic":
        customers, macro = generate_all(SynthConfig(num_customers=args.customers, months=args.months, seed=args.seed))
    elif args.data_source=="parquet":
        customers, macro = load_from_parquet(args.parquet_dir)
    else:
        customers, macro = load_from_hive(args.hive_db)

    env_cfg = EnvConfig(history_len=6, months=args.months, seed=args.seed)
    base_policy = baseline_rule_factory(env_cfg.history_len)
    base_panel = simulate_accounts(customers, macro, env_cfg, policy_fn=base_policy, seed=args.seed+999)
    base_m, base_sum = compute_metrics(base_panel)

    model = PPO.load(Path(args.model))
    vec_stats = load_vec_stats(Path("outputs/models/vec_stats.npz"))
    rl_policy = rl_policy_fn_factory(model, vec_stats)
    rl_panel = simulate_accounts(customers, macro, env_cfg, policy_fn=rl_policy, seed=args.seed+202)
    rl_m, rl_sum = compute_metrics(rl_panel)

    sns.set(style="whitegrid")
    plt.figure(); plt.plot(base_m["month"], base_m["cum_profit"], label="Baseline"); plt.plot(rl_m["month"], rl_m["cum_profit"], label="RL")
    plt.title("Cumulative Profit"); plt.xlabel("Month"); plt.ylabel("Cumulative Profit"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"cumulative_profit.png"); plt.close()

    plt.figure(); plt.plot(base_m["month"], base_m["defaults"], label="Baseline"); plt.plot(rl_m["month"], rl_m["defaults"], label="RL")
    plt.title("Monthly Defaults"); plt.xlabel("Month"); plt.ylabel("Defaults"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"default_trend.png"); plt.close()

    plt.figure(); plt.plot(base_m["month"], base_m["avg_limit"], label="Baseline"); plt.plot(rl_m["month"], rl_m["avg_limit"], label="RL")
    plt.title("Average Credit Limit"); plt.xlabel("Month"); plt.ylabel("Average Limit"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir/"avg_limit.png"); plt.close()

    print("\n=== Final Metrics ===")
    print(f"Baseline total profit: {base_sum['total_profit']:.2f}, defaults: {base_sum['total_defaults']}")
    print(f"RL       total profit: {rl_sum['total_profit']:.2f}, defaults: {rl_sum['total_defaults']}")
    print(f"% improvement (profit): {pct_improve(rl_sum['total_profit'], base_sum['total_profit']):.2f}%")
    print(f"Baseline total reward: {base_sum['total_reward']:.2f}")
    print(f"RL       total reward: {rl_sum['total_reward']:.2f}")
    print(f"% improvement (reward): {pct_improve(rl_sum['total_reward'], base_sum['total_reward']):.2f}%")
    print(f"Baseline avg limit: {base_sum['avg_limit']:.2f}")
    print(f"RL       avg limit: {rl_sum['avg_limit']:.2f}")
    print(f"Baseline avg default prob: {base_sum['avg_default_prob']:.4f}")
    print(f"RL       avg default prob: {rl_sum['avg_default_prob']:.4f}")

if __name__ == "__main__":
    main()
