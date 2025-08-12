from __future__ import annotations
import numpy as np, pandas as pd
from collections import defaultdict, deque
from credit_limit_env import simulate_accounts, EnvConfig
from synthetic_data import SynthConfig, generate_all

def baseline_rule_factory(history_len: int = 6):
    util_hist = defaultdict(lambda: deque(maxlen=history_len))
    repay_hist = defaultdict(lambda: deque(maxlen=history_len))
    def policy_fn(state: dict) -> int:
        util = float(state["util"]); repay = float(state["repay_rate"]); key=0
        util_hist[key].append(util); repay_hist[key].append(repay)
        util_ok = (len(util_hist[key])>=history_len) and (np.mean(util_hist[key])<0.50)
        missed = sum(1 for r in repay_hist[key] if r < 0.1)
        if missed > 2: return 1  # -10%
        elif util_ok:  return 3  # +10%
        else:          return 2  # keep
    return policy_fn

def run_baseline(num_customers=1000, months=60, seed=7) -> pd.DataFrame:
    customers, macro = generate_all(SynthConfig(num_customers=num_customers, months=months, seed=seed))
    cfg = EnvConfig(history_len=6, months=months, seed=seed)
    policy = baseline_rule_factory(cfg.history_len)
    return simulate_accounts(customers, macro, cfg, policy_fn=policy, seed=seed+999)

if __name__ == "__main__":
    res = run_baseline()
    print(res.head()); print("rows:", len(res))
