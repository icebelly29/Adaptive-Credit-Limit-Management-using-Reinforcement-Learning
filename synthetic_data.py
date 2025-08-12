from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd

@dataclass
class SynthConfig:
    num_customers: int = 1000
    months: int = 60
    seed: int = 7

def generate_synthetic_customers(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    patterns = rng.choice(["normal", "high", "seasonal"], size=cfg.num_customers, p=[0.55,0.25,0.20])
    behaviors = rng.choice(["on_time", "late", "partial"], size=cfg.num_customers, p=[0.55,0.25,0.20])

    base_spend, spend_std = [], []
    for p in patterns:
        if p=="normal": base_spend.append(rng.uniform(500,1200)); spend_std.append(rng.uniform(80,200))
        elif p=="high": base_spend.append(rng.uniform(1200,2500)); spend_std.append(rng.uniform(150,350))
        else:           base_spend.append(rng.uniform(600,1500)); spend_std.append(rng.uniform(100,250))

    base_repay_mean, base_repay_std = [], []
    for b in behaviors:
        if b=="on_time": base_repay_mean.append(rng.uniform(0.85,0.98)); base_repay_std.append(rng.uniform(0.02,0.07))
        elif b=="late":  base_repay_mean.append(rng.uniform(0.55,0.8));  base_repay_std.append(rng.uniform(0.05,0.12))
        else:            base_repay_mean.append(rng.uniform(0.25,0.55)); base_repay_std.append(rng.uniform(0.07,0.15))

    risk = rng.beta(2,5, size=cfg.num_customers)
    init_limit = []
    for s, r in zip(base_spend, risk):
        base = s * rng.uniform(2.5, 4.0)
        adj = base * (1.2 - 0.8*r)
        init_limit.append(np.clip(adj, 500, 20000))

    return pd.DataFrame(dict(
        customer_id=np.arange(cfg.num_customers, dtype=int),
        pattern=patterns, repay_behavior=behaviors,
        base_spend=np.array(base_spend), spend_std=np.array(spend_std),
        base_repay_mean=np.array(base_repay_mean), base_repay_std=np.array(base_repay_std),
        risk_score=risk.astype(float), init_limit=np.array(init_limit)
    ))

def generate_macro_series(cfg: SynthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed+101)
    T = cfg.months; rates = np.zeros(T); unemp = np.zeros(T)
    rates[0]=unemp[0]=0.05
    for t in range(1,T):
        rates[t]=np.clip(rates[t-1]+rng.normal(0,0.002)-0.05*(rates[t-1]-0.05),0.0,0.25)
        unemp[t]=np.clip(unemp[t-1]+rng.normal(0,0.003)-0.1*(unemp[t-1]-0.05),0.02,0.20)
    if T>24:
        rates[24]=min(0.30, rates[24]+0.02); unemp[24]=min(0.35, unemp[24]+0.03)
    return pd.DataFrame({"month":np.arange(T), "interest_rate":rates, "unemployment_rate":unemp})

def generate_all(cfg: SynthConfig):
    return generate_synthetic_customers(cfg), generate_macro_series(cfg)
