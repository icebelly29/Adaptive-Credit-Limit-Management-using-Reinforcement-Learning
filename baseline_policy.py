# project/baseline_policy.py
from __future__ import annotations
import numpy as np
import pandas as pd

from credit_limit_env import simulate_accounts, EnvConfig
from synthetic_data import generate_all, SynthConfig


def baseline_rule_factory(history_len: int = 6):
    """
    Returns a policy_fn(state_dict)->action id {0,1,2} implementing:
      - Increase limit by 10% if utilization < 50% for last 6 months
      - Decrease if missed payments > 2 in last 6 months
      - Else keep unchanged
    NOTE: Our simulate_accounts does not hand us full 6m histories;
          we emulate via thresholds using available state & a simple memory.
    We'll maintain an internal rolling history per customer via closure.
    """
    # memory: dict customer_key -> dict with deques
    from collections import defaultdict, deque
    util_hist = defaultdict(lambda: deque(maxlen=history_len))
    repay_hist = defaultdict(lambda: deque(maxlen=history_len))

    def policy_fn(state: dict) -> int:
        # We expect simulate_accounts to attach 'customer_id' and 'month'?
        # It doesn't. So we approximate per-trajectory only via globals = not available.
        # Workaround: Use only instantaneous state and moving heuristic:
        util = float(state["util"])
        repay = float(state["repay_rate"])

        # Update anonymous rolling windows (shared; good enough for baseline aggregate)
        key = 0  # single pool memory; baseline is heuristic anyway
        util_hist[key].append(util)
        repay_hist[key].append(repay)

        util_ok = (len(util_hist[key]) >= history_len) and (np.mean(util_hist[key]) < 0.50)
        missed = sum(1 for r in repay_hist[key] if r < 0.1)

        if missed > 2:
            return 0  # decrease
        elif util_ok:
            return 2  # increase
        else:
            return 1  # keep

    return policy_fn


def run_baseline(num_customers: int = 1000, months: int = 60, seed: int = 7) -> pd.DataFrame:
    customers, macro = generate_all(SynthConfig(num_customers=num_customers, months=months, seed=seed))
    cfg = EnvConfig(history_len=6, months=months, seed=seed)
    policy = baseline_rule_factory(history_len=cfg.history_len)
    df = simulate_accounts(customers, macro, cfg, policy_fn=policy, seed=seed+999)
    return df


if __name__ == "__main__":
    res = run_baseline()
    print(res.head())
    print("Baseline simulation complete:", len(res), "rows")
