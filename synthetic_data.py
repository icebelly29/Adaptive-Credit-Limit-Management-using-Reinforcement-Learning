# project/synthetic_data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class SynthConfig:
    num_customers: int = 1000
    months: int = 60
    seed: int = 7
    # base levels
    base_interest: float = 0.05      # 5% annual short rate
    base_unemployment: float = 0.05  # 5% unemployment
    shock_enabled: bool = False
    shock_month: int = 24
    shock_unemp_jump: float = 0.03   # +3% absolute
    shock_rate_jump: float = 0.02    # +2% absolute


def generate_synthetic_customers(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # Spending patterns
    patterns = rng.choice(["normal", "high", "seasonal"], size=cfg.num_customers, p=[0.55, 0.25, 0.20])

    # Repayment behaviors
    behaviors = rng.choice(["on_time", "late", "partial"], size=cfg.num_customers, p=[0.55, 0.25, 0.20])

    # Base spend mean & std by pattern
    base_spend = []
    spend_std = []
    for p in patterns:
        if p == "normal":
            base_spend.append(rng.uniform(500, 1200))
            spend_std.append(rng.uniform(80, 200))
        elif p == "high":
            base_spend.append(rng.uniform(1200, 2500))
            spend_std.append(rng.uniform(150, 350))
        else:  # seasonal
            base_spend.append(rng.uniform(600, 1500))
            spend_std.append(rng.uniform(100, 250))

    # Repayment base stats by behavior
    base_repay_mean = []
    base_repay_std = []
    for b in behaviors:
        if b == "on_time":
            base_repay_mean.append(rng.uniform(0.85, 0.98))
            base_repay_std.append(rng.uniform(0.02, 0.07))
        elif b == "late":
            base_repay_mean.append(rng.uniform(0.55, 0.8))
            base_repay_std.append(rng.uniform(0.05, 0.12))
        else:  # partial
            base_repay_mean.append(rng.uniform(0.25, 0.55))
            base_repay_std.append(rng.uniform(0.07, 0.15))

    # Credit risk score (0..1)
    risk_score = rng.beta(2, 5, size=cfg.num_customers)  # skewed to safer customers

    # Initial limits informed by spend & risk
    init_limit = []
    for s, r in zip(base_spend, risk_score):
        # safer & higher spenders start with higher limits
        base = s * rng.uniform(2.5, 4.0)
        adj = base * (1.2 - 0.8 * r)  # reduce for risky
        init_limit.append(np.clip(adj, 500, 20000))

    customers = pd.DataFrame({
        "customer_id": np.arange(cfg.num_customers, dtype=int),
        "pattern": patterns,
        "repay_behavior": behaviors,
        "base_spend": np.array(base_spend, dtype=float),
        "spend_std": np.array(spend_std, dtype=float),
        "base_repay_mean": np.array(base_repay_mean, dtype=float),
        "base_repay_std": np.array(base_repay_std, dtype=float),
        "risk_score": risk_score.astype(float),
        "init_limit": np.array(init_limit, dtype=float),
    })
    return customers


def generate_macro_series(cfg: SynthConfig) -> pd.DataFrame:
    """
    Random walk macro with optional one-off recession shock and mild mean reversion.
    interest_rate, unemployment_rate as decimals.
    """
    rng = np.random.default_rng(cfg.seed + 101)
    T = cfg.months
    rates = np.zeros(T, dtype=float)
    unemp = np.zeros(T, dtype=float)
    rates[0] = cfg.base_interest
    unemp[0] = cfg.base_unemployment

    for t in range(1, T):
        rates[t] = rates[t-1] + rng.normal(0, 0.002) - 0.05 * (rates[t-1] - cfg.base_interest)
        unemp[t] = unemp[t-1] + rng.normal(0, 0.003) - 0.1 * (unemp[t-1] - cfg.base_unemployment)
        rates[t] = float(np.clip(rates[t], 0.0, 0.25))
        unemp[t] = float(np.clip(unemp[t], 0.02, 0.20))

    if cfg.shock_enabled and 0 <= cfg.shock_month < T:
        # Apply spike, then smooth reversion over next 6 months
        sm = cfg.shock_month
        rates[sm:] = np.clip(rates[sm:] + cfg.shock_rate_jump, 0.0, 0.3)
        unemp[sm:] = np.clip(unemp[sm:] + cfg.shock_unemp_jump, 0.0, 0.35)
        for k in range(1, 7):
            t = sm + k
            if t < T:
                rates[t] = rates[t] - (k/7.0) * cfg.shock_rate_jump * 0.6
                unemp[t] = unemp[t] - (k/7.0) * cfg.shock_unemp_jump * 0.6
                rates[t] = float(np.clip(rates[t], 0.0, 0.3))
                unemp[t] = float(np.clip(unemp[t], 0.0, 0.35))

    return pd.DataFrame({
        "month": np.arange(T, dtype=int),
        "interest_rate": rates,
        "unemployment_rate": unemp,
    })


def generate_all(cfg: SynthConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return generate_synthetic_customers(cfg), generate_macro_series(cfg)
