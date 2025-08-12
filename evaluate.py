from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO

from credit_limit_env import simulate_accounts, EnvConfig
from synthetic_data import SynthConfig, generate_all
from baseline_policy import baseline_rule_factory


def load_vec_stats(stats_path: Path):
    if stats_path.exists():
        data = np.load(stats_path)
        mean = data["mean"]
        var = data["var"]
        clip_obs = float(data["clip_obs"])
        return dict(mean=mean, var=var, clip_obs=clip_obs)
    return None


def rl_policy_fn_factory(model: PPO, vec_stats: dict | None):
    """
    Wrap a trained PPO model to act as a policy_fn for simulate_accounts.
    If vec_stats is provided, normalize obs to match training VecNormalize.
    """
    def policy_fn(state: dict) -> int:
        obs = np.array([
            state["limit_norm"],
            state["avg_spend_norm"],
            state["repay_rate"],
            state["util"],
            state["interest_rate"],
            state["unemployment_rate"],
        ], dtype=np.float32).reshape(1, -1)

        if vec_stats is not None:
            mean = vec_stats["mean"]
            var = vec_stats["var"]
            clip_obs = vec_stats["clip_obs"]
            obs = (obs - mean) / np.sqrt(var + 1e-8)
            obs = np.clip(obs, -clip_obs, clip_obs)

        action, _ = model.predict(obs, deterministic=True)
        return int(action[0])

    return policy_fn


def compute_metrics(panel: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    panel = panel.copy()
    panel["net_reward"] = panel["reward"]
    # Aggregate by month
    by_m = panel.groupby("month").agg(
        cum_profit=("profit", "sum"),
        cum_reward=("net_reward", "sum"),
        defaults=("default", "sum"),
        avg_limit=("limit", "mean"),
        avg_defprob=("default_prob", "mean"),
    ).reset_index()
    by_m["cum_profit_cumsum"] = by_m["cum_profit"].cumsum()
    by_m["cum_reward_cumsum"] = by_m["cum_reward"].cumsum()

    # Final summary
    summary = dict(
        total_profit=float(panel["profit"].sum()),
        total_reward=float(panel["reward"].sum()),
        total_defaults=int(panel["default"].sum()),
        avg_limit=float(panel["limit"].mean()),
        avg_default_prob=float(panel["default_prob"].mean()),
    )
    return by_m, summary


def plot_all(baseline_m: pd.DataFrame, rl_m: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # Cumulative profit
    plt.figure()
    plt.plot(baseline_m["month"], baseline_m["cum_profit_cumsum"], label="Baseline")
    plt.plot(rl_m["month"], rl_m["cum_profit_cumsum"], label="RL")
    plt.title("Cumulative Profit over Time")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Profit")
    plt.legend()
    plt.tight_layout()
    p1 = outdir / "cumulative_profit.png"
    plt.savefig(p1)
    plt.close()

    # Default count per month (trend)
    plt.figure()
    plt.plot(baseline_m["month"], baseline_m["defaults"], label="Baseline")
    plt.plot(rl_m["month"], rl_m["defaults"], label="RL")
    plt.title("Monthly Defaults")
    plt.xlabel("Month")
    plt.ylabel("Defaults")
    plt.legend()
    plt.tight_layout()
    p2 = outdir / "default_trend.png"
    plt.savefig(p2)
    plt.close()

    # Average credit limits
    plt.figure()
    plt.plot(baseline_m["month"], baseline_m["avg_limit"], label="Baseline")
    plt.plot(rl_m["month"], rl_m["avg_limit"], label="RL")
    plt.title("Average Credit Limit")
    plt.xlabel("Month")
    plt.ylabel("Average Limit")
    plt.legend()
    plt.tight_layout()
    p3 = outdir / "avg_limit.png"
    plt.savefig(p3)
    plt.close()

    print(f"Saved plots:\n- {p1}\n- {p2}\n- {p3}")


def percent_improvement(rl: float, base: float) -> float:
    if base == 0:
        return np.inf if rl > 0 else 0.0
    return 100.0 * (rl - base) / abs(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/ppo_credit_limit.zip")
    parser.add_argument("--customers", type=int, default=1000)
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shock", action="store_true", help="Enable macro shock during evaluation")
    args = parser.parse_args()

    outdir = Path("outputs/eval")
    outdir.mkdir(parents=True, exist_ok=True)

    # Data
    synth_cfg = SynthConfig(
        num_customers=args.customers,
        months=args.months,
        seed=args.seed,
        shock_enabled=args.shock,
        shock_month=24,
    )
    customers, macro = generate_all(synth_cfg)
    env_cfg = EnvConfig(history_len=6, months=args.months, seed=args.seed)

    # Baseline simulation
    baseline_policy = baseline_rule_factory(history_len=env_cfg.history_len)
    base_panel = simulate_accounts(customers, macro, env_cfg, policy_fn=baseline_policy, seed=args.seed + 999)
    base_m, base_sum = compute_metrics(base_panel)

    # RL model + normalization stats
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_rl_agent.py first.")
    model = PPO.load(model_path)

    vec_stats = load_vec_stats(Path("outputs/models/vec_stats.npz"))
    rl_policy = rl_policy_fn_factory(model, vec_stats)
    rl_panel = simulate_accounts(customers, macro, env_cfg, policy_fn=rl_policy, seed=args.seed + 202)
    rl_m, rl_sum = compute_metrics(rl_panel)

    # Plots
    plot_all(base_m, rl_m, outdir)

    # Print key stats
    print("\n=== Final Metrics ===")
    print(f"Baseline total profit: {base_sum['total_profit']:.2f}, defaults: {base_sum['total_defaults']}")
    print(f"RL       total profit: {rl_sum['total_profit']:.2f}, defaults: {rl_sum['total_defaults']}")
    print(f"% improvement (profit): {percent_improvement(rl_sum['total_profit'], base_sum['total_profit']):.2f}%")

    print(f"Baseline total reward: {base_sum['total_reward']:.2f}")
    print(f"RL       total reward: {rl_sum['total_reward']:.2f}")
    print(f"% improvement (reward): {percent_improvement(rl_sum['total_reward'], base_sum['total_reward']):.2f}%")

    print(f"Baseline avg limit: {base_sum['avg_limit']:.2f}")
    print(f"RL       avg limit: {rl_sum['avg_limit']:.2f}")
    print(f"Baseline avg default prob: {base_sum['avg_default_prob']:.4f}")
    print(f"RL       avg default prob: {rl_sum['avg_default_prob']:.4f}")


if __name__ == "__main__":
    main()
import numpy as np