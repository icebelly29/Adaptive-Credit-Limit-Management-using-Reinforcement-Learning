from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
try:
    import gymnasium as gym
except ImportError:
    import gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    history_len: int = 6
    months: int = 60
    min_limit: float = 500.0
    max_limit: float = 20000.0
    util_cap: float = 0.98
    interchange_fee: float = 0.02
    margin_over_rate: float = 0.15
    lgd: float = 0.85
    overexposure_lambda: float = 8.0
    default_penalty: float = 800.0
    limit_scale: float = 10000.0
    risk_penalty_lambda: float = 1.0
    capital_cost_annual: float = 0.04
    undrawn_weight: float = 1.0
    seed: int = 42


class CreditLimitEnv(gym.Env):
    """
    Observation (6):
      [limit_norm, avg_spend_norm, last_repay_rate, util, interest_rate, unemployment_rate]
    Action (Discrete 5): {0:-20%, 1:-10%, 2:0, 3:+10%, 4:+20%}
    Reward: profit (interest+fee) - capital/line cost - risk penalty - overexposure - default costs
    """
    metadata = {"render_modes": []}

    def __init__(self, customers_df: pd.DataFrame, macro_df: pd.DataFrame, config: Optional[EnvConfig] = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.customers = customers_df.reset_index(drop=True).copy()
        self.macro = macro_df.reset_index(drop=True).copy()
        self.T = min(self.cfg.months, len(self.macro))

        self.action_space = spaces.Discrete(5)
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([5, 5, 1, 1, 0.5, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.cur_month = 0
        self.customer_idx = -1
        self.balance = 0.0
        self.limit = 0.0
        self.last_repay_rate = 1.0
        self.spend_hist: list[float] = []
        self.repay_hist: list[float] = []
        self.missed_hist: list[int] = []
        self.done = False
        self.info_last: Dict[str, Any] = {}

    # --- helpers ---
    def _get_customer_row(self) -> pd.Series:
        return self.customers.iloc[self.customer_idx]

    def _macro_at(self, t: int) -> Tuple[float, float]:
        row = self.macro.iloc[t]
        return float(row["interest_rate"]), float(row["unemployment_rate"])

    def _seasonal_multiplier(self, pattern: str, t: int) -> float:
        return 1.0 + 0.2 * math.sin(2 * math.pi * (t % 12) / 12.0) if pattern == "seasonal" else 1.0

    def _draw_spend(self, base_mean: float, std: float, pattern: str, t: int) -> float:
        seasonal = self._seasonal_multiplier(pattern, t)
        return float(max(self.rng.normal(loc=base_mean * seasonal, scale=std), 0.0))

    def _draw_repay_rate(self, behavior: str, base_mean: float, base_std: float, unemp: float) -> float:
        mean_adj = float(np.clip(base_mean - 0.3 * (unemp - 0.05), 0.05, 1.05))
        rr = self.rng.normal(loc=mean_adj, scale=base_std)
        if behavior == "on_time": rr = max(rr, 0.7)
        elif behavior == "late":  rr = np.clip(rr, 0.3, 0.9)
        else:                     rr = np.clip(rr, 0.1, 0.7)
        return float(np.clip(rr, 0.0, 1.0))

    def _default_probability(self, util: float, repay_rate: float, unemp: float, int_rate: float, risk_score: float) -> float:
        b0, b_util, b_repay, b_unemp, b_rate, b_risk = -4.0, 2.5, -3.0, 6.0, 1.5, 3.0
        x = b0 + b_util*util + b_repay*repay_rate + b_unemp*unemp + b_rate*int_rate + b_risk*risk_score
        p = 1.0 / (1.0 + math.exp(-x))
        return float(np.clip(p, 0.001, 0.8))

    def _interchange(self, spend: float) -> float:
        return self.cfg.interchange_fee * spend

    def _monthly_rate(self, macro_interest_annual: float) -> float:
        return float(np.clip(macro_interest_annual + self.cfg.margin_over_rate, 0.0, 0.8) / 12.0)

    def _overexposure_penalty(self, limit: float, avg_spend: float, def_prob: float) -> float:
        if def_prob <= 0.15 or limit <= 0: return 0.0
        excess = max(limit - 2.0 * avg_spend, 0.0) / max(limit, 1e-6)
        return self.cfg.overexposure_lambda * excess

    def _obs(self, int_rate: float, unemp: float) -> np.ndarray:
        avg_spend = np.mean(self.spend_hist[-self.cfg.history_len:]) if self.spend_hist else 0.0
        util = np.clip(self.balance / max(self.limit, 1.0), 0.0, 1.0)
        return np.array([
            self.limit / self.cfg.limit_scale,
            avg_spend / self.cfg.limit_scale,
            np.clip(self.last_repay_rate, 0, 1),
            util,
            np.clip(int_rate, 0, 0.5),
            np.clip(unemp, 0, 0.5),
        ], dtype=np.float32)

    # --- gym api ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.done = False; self.cur_month = 0
        self.customer_idx = self.rng.integers(low=0, high=len(self.customers))
        row = self._get_customer_row()
        self.limit = float(np.clip(float(row["init_limit"]), self.cfg.min_limit, self.cfg.max_limit))
        self.balance = 0.0; self.last_repay_rate = 1.0
        self.spend_hist, self.repay_hist, self.missed_hist = [], [], []
        int_rate, unemp = self._macro_at(0)
        for _ in range(self.cfg.history_len):
            s = self._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], 0)
            rr = self._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp)
            self.spend_hist.append(s); self.repay_hist.append(rr); self.missed_hist.append(1 if rr < 0.1 else 0)
        return self._obs(int_rate, unemp), {}

    def step(self, action: int):
        if self.done: raise RuntimeError("Call reset() before step() after episode is done.")
        action = int(action); row = self._get_customer_row()
        int_rate, unemp = self._macro_at(self.cur_month)

        delta_map = {0: -0.20, 1: -0.10, 2: 0.0, 3: 0.10, 4: 0.20}
        self.limit *= (1.0 + delta_map.get(action, 0.0))
        self.limit = float(np.clip(self.limit, self.cfg.min_limit, self.cfg.max_limit))

        spend = self._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], self.cur_month)
        spend = float(np.clip(spend, 0.0, self.limit * self.cfg.util_cap))
        self.balance += spend

        repay_rate = self._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp)
        payment = float(np.clip(repay_rate * self.balance, 0.0, self.balance))
        self.balance -= payment

        util = np.clip(self.balance / max(self.limit, 1.0), 0.0, 1.0)
        def_prob = self._default_probability(util, repay_rate, unemp, int_rate, row["risk_score"])
        default_flag = self.rng.uniform() < def_prob

        monthly_rate = self._monthly_rate(int_rate)
        interest = monthly_rate * self.balance
        fee = self._interchange(spend)
        profit = interest + fee
        reward = profit

        if default_flag:
            loss = self.cfg.lgd * self.balance
            reward -= (loss + self.cfg.default_penalty)
            self.balance = 0.0

        cap_rate_m = self.cfg.capital_cost_annual / 12.0
        undrawn = max(self.limit - self.balance, 0.0)
        cap_base = self.cfg.undrawn_weight * undrawn + (1 - self.cfg.undrawn_weight) * self.limit
        reward -= cap_rate_m * cap_base

        reward -= self.cfg.risk_penalty_lambda * def_prob
        avg_spend = float(np.mean(self.spend_hist[-self.cfg.history_len:])) if self.spend_hist else spend
        reward -= self._overexposure_penalty(self.limit, avg_spend, def_prob)

        self.spend_hist.append(spend); self.repay_hist.append(repay_rate)
        self.missed_hist.append(1 if repay_rate < 0.1 else 0)
        self.last_repay_rate = repay_rate
        self.cur_month += 1
        terminated = bool(default_flag or (self.cur_month >= self.T))
        self.done = terminated
        info = dict(
            customer_id=int(row["customer_id"]), spend=spend, payment=payment, repay_rate=repay_rate,
            interest=interest, fee=fee, profit=profit, default_prob=def_prob, default=default_flag,
            balance=self.balance, limit=self.limit, month=self.cur_month,
            macro_interest=int_rate, macro_unemployment=unemp, action=action,
        )
        return self._obs(int_rate, unemp), float(reward), terminated, False, info


def simulate_accounts(customers_df: pd.DataFrame, macro_df: pd.DataFrame, config: Optional[EnvConfig], policy_fn, seed: int = 123) -> pd.DataFrame:
    cfg = config or EnvConfig()
    rng = np.random.default_rng(seed)
    rows = []
    env = CreditLimitEnv(customers_df, macro_df, cfg)

    for i in range(len(customers_df)):
        env.customer_idx = i; env.cur_month = 0; env.done = False
        row = env._get_customer_row()
        env.limit = float(np.clip(row["init_limit"], cfg.min_limit, cfg.max_limit))
        env.balance = 0.0; env.last_repay_rate = 1.0
        env.spend_hist, env.repay_hist, env.missed_hist = [], [], []
        int0, unemp0 = env._macro_at(0)
        for _ in range(cfg.history_len):
            s0 = env._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], 0)
            rr0 = env._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp0)
            env.spend_hist.append(s0); env.repay_hist.append(rr0); env.missed_hist.append(1 if rr0 < 0.1 else 0)

        for _ in range(env.T):
            if env.done: break
            int_rate, unemp = env._macro_at(env.cur_month)
            state = {
                "limit_norm": env.limit / cfg.limit_scale,
                "avg_spend_norm": (np.mean(env.spend_hist[-cfg.history_len:]) / cfg.limit_scale) if env.spend_hist else 0.0,
                "repay_rate": env.last_repay_rate,
                "util": float(np.clip(env.balance / max(env.limit, 1.0), 0.0, 1.0)),
                "interest_rate": float(int_rate),
                "unemployment_rate": float(unemp),
            }
            action = int(policy_fn(state))
            delta_map = {0: -0.20, 1: -0.10, 2: 0.0, 3: 0.10, 4: 0.20}
            env.limit *= (1.0 + delta_map.get(action, 0.0))
            env.limit = float(np.clip(env.limit, cfg.min_limit, cfg.max_limit))

            spend = env._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], env.cur_month)
            spend = float(np.clip(spend, 0.0, env.limit * cfg.util_cap))
            env.balance += spend

            repay_rate = env._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp)
            payment = float(np.clip(repay_rate * env.balance, 0.0, env.balance))
            env.balance -= payment

            util = np.clip(env.balance / max(env.limit, 1.0), 0.0, 1.0)
            def_prob = env._default_probability(util, repay_rate, unemp, int_rate, row["risk_score"])
            default_flag = rng.uniform() < def_prob

            monthly_rate = env._monthly_rate(int_rate)
            interest = monthly_rate * env.balance
            fee = env._interchange(spend)
            profit = interest + fee
            reward = profit

            if default_flag:
                loss = cfg.lgd * env.balance
                reward -= (loss + cfg.default_penalty)
                env.balance = 0.0

            cap_rate_m = cfg.capital_cost_annual / 12.0
            undrawn = max(env.limit - env.balance, 0.0)
            cap_base = cfg.undrawn_weight * undrawn + (1 - cfg.undrawn_weight) * env.limit
            reward -= cap_rate_m * cap_base

            reward -= cfg.risk_penalty_lambda * def_prob
            avg_spend = float(np.mean(env.spend_hist[-cfg.history_len:])) if env.spend_hist else spend
            reward -= env._overexposure_penalty(env.limit, avg_spend, def_prob)

            env.spend_hist.append(spend); env.repay_hist.append(repay_rate)
            env.missed_hist.append(1 if repay_rate < 0.1 else 0)
            env.last_repay_rate = repay_rate

            env.cur_month += 1
            env.done = bool(default_flag or (env.cur_month >= env.T))
            rows.append(dict(
                customer_id=int(row["customer_id"]), month=int(env.cur_month), action=action,
                limit=float(env.limit), spend=float(spend), payment=float(payment),
                repay_rate=float(repay_rate), interest=float(interest), fee=float(fee),
                profit=float(profit), reward=float(reward),
                default_prob=float(def_prob), default=int(default_flag),
                balance=float(env.balance), macro_interest=float(int_rate), macro_unemployment=float(unemp),
            ))

    return pd.DataFrame(rows)
