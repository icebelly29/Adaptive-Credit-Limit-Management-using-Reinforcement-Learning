from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# Prefer Gymnasium (SB3 v2+), fallback to gym if needed.
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

from gymnasium import spaces


@dataclass
class EnvConfig:
    history_len: int = 6              # rolling window for avg spend, missed payments
    months: int = 48                  # episode horizon in months (trainer sets 60)
    min_limit: float = 500.0
    max_limit: float = 20000.0
    util_cap: float = 0.98            # cap utilization so balances don't explode
    interchange_fee: float = 0.02     # % of spend
    margin_over_rate: float = 0.15    # issuer APR margin over macro interest (annual)
    lgd: float = 0.85                 # loss given default fraction of balance
    overexposure_lambda: float = 8.0  # penalty weight for excessive limit vs spend
    default_penalty: float = 800.0    # fixed penalty added on default
    limit_scale: float = 10000.0      # normalization constant
    risk_penalty_lambda: float = 2.0  # per-step penalty on default probability
    seed: int = 42


class CreditLimitEnv(gym.Env):
    """
    One episode = one customer over 'months' timesteps (monthly cadence).

    Observation (6):
      0: current credit limit (normalized by limit_scale)
      1: avg monthly spend last N months (normalized)
      2: last repayment rate (0..1)
      3: utilization ratio current month (0..1)
      4: macro interest rate (annual, 0..1)
      5: macro unemployment rate (0..1)

    Action (Discrete 3):
      0: decrease limit by 10%
      1: keep unchanged
      2: increase limit by 10%

    Reward:
      profit (interest + interchange) - risk penalty - overexposure penalty - default costs
    Episode ends on default or horizon.
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        customers_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        config: Optional[EnvConfig] = None,
    ):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # Data
        self.customers = customers_df.reset_index(drop=True).copy()
        self.macro = macro_df.reset_index(drop=True).copy()
        self.T = min(self.cfg.months, len(self.macro))

        # Action/Obs spaces
        self.action_space = spaces.Discrete(3)
        # [limit_norm, avg_spend_norm, repay_rate, util, int_rate, unemp]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([5.0, 5.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float32)  # generous caps
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Episode state vars
        self.cur_month = 0
        self.customer_idx: int = -1
        self.balance: float = 0.0
        self.limit: float = 0.0
        self.last_repay_rate: float = 1.0
        self.spend_hist: list[float] = []
        self.repay_hist: list[float] = []
        self.missed_hist: list[int] = []
        self.done: bool = False

        # For logging
        self.info_last: Dict[str, Any] = {}

    # ------------ Core simulation helpers ------------ #

    def _get_customer_row(self) -> pd.Series:
        return self.customers.iloc[self.customer_idx]

    def _macro_at(self, t: int) -> Tuple[float, float]:
        row = self.macro.iloc[t]
        return float(row["interest_rate"]), float(row["unemployment_rate"])

    def _seasonal_multiplier(self, pattern: str, t: int) -> float:
        # Simple mild seasonal bump for 'seasonal'
        if pattern != "seasonal":
            return 1.0
        # 12-month periodic sin wave in [0.8, 1.2]
        return 1.0 + 0.2 * math.sin(2 * math.pi * (t % 12) / 12.0)

    def _draw_spend(self, base_mean: float, std: float, pattern: str, t: int) -> float:
        seasonal = self._seasonal_multiplier(pattern, t)
        spend = self.rng.normal(loc=base_mean * seasonal, scale=std)
        return float(max(spend, 0.0))

    def _draw_repay_rate(self, behavior: str, base_mean: float, base_std: float, unemp: float) -> float:
        # Higher unemployment drags repayment
        mean_adj = base_mean - 0.3 * (unemp - 0.05)  # centered around 5% unemployment
        mean_adj = float(np.clip(mean_adj, 0.05, 1.05))
        rr = self.rng.normal(loc=mean_adj, scale=base_std)
        # behavior skews:
        if behavior == "on_time":
            rr = max(rr, 0.7)
        elif behavior == "late":
            rr = np.clip(rr, 0.3, 0.9)
        else:  # partial
            rr = np.clip(rr, 0.1, 0.7)
        return float(np.clip(rr, 0.0, 1.0))

    def _default_probability(self, util: float, repay_rate: float, unemp: float, int_rate: float, risk_score: float) -> float:
        """
        Logistic default model: higher with high util, low repay, high unemployment & rates, high risk.
        risk_score in [0,1] (0 low risk, 1 high risk)
        """
        # Coefficients (tunable)
        b0 = -4.0
        b_util = 2.5
        b_repay = -3.0
        b_unemp = 6.0
        b_rate = 1.5
        b_risk = 3.0

        x = (b0
             + b_util * util
             + b_repay * repay_rate
             + b_unemp * unemp
             + b_rate * int_rate
             + b_risk * risk_score)
        p = 1.0 / (1.0 + math.exp(-x))
        return float(np.clip(p, 0.001, 0.8))  # cap extremes

    def _interchange(self, spend: float) -> float:
        return self.cfg.interchange_fee * spend

    def _monthly_rate(self, macro_interest_annual: float) -> float:
        apr = np.clip(macro_interest_annual + self.cfg.margin_over_rate, 0.0, 0.8)
        return float(apr / 12.0)

    def _overexposure_penalty(self, limit: float, avg_spend: float, def_prob: float) -> float:
        # penalize big limit when risk is elevated
        if def_prob <= 0.15 or limit <= 0:
            return 0.0
        excess = max(limit - 2.0 * avg_spend, 0.0) / max(limit, 1e-6)
        return self.cfg.overexposure_lambda * excess

    def _obs(self, int_rate: float, unemp: float) -> np.ndarray:
        avg_spend = np.mean(self.spend_hist[-self.cfg.history_len:]) if self.spend_hist else 0.0
        util = np.clip(self.balance / max(self.limit, 1.0), 0.0, 1.0)
        obs = np.array([
            self.limit / self.cfg.limit_scale,
            avg_spend / self.cfg.limit_scale,
            np.clip(self.last_repay_rate, 0.0, 1.0),
            util,
            np.clip(int_rate, 0.0, 0.5),
            np.clip(unemp, 0.0, 0.5),
        ], dtype=np.float32)
        return obs

    # ------------ Gym API ------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.done = False
        self.cur_month = 0
        self.customer_idx = self.rng.integers(low=0, high=len(self.customers))
        row = self._get_customer_row()

        self.limit = float(row["init_limit"])
        self.limit = float(np.clip(self.limit, self.cfg.min_limit, self.cfg.max_limit))
        self.balance = 0.0
        self.last_repay_rate = 1.0
        self.spend_hist, self.repay_hist, self.missed_hist = [], [], []

        # Warm-up synthetic history for state (does not affect rewards)
        int_rate, unemp = self._macro_at(0)
        for _ in range(self.cfg.history_len):
            spend = self._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], 0)
            rr = self._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp)
            self.spend_hist.append(spend)
            self.repay_hist.append(rr)
            self.missed_hist.append(1 if rr < 0.1 else 0)

        obs = self._obs(int_rate, unemp)
        self.info_last = {"customer_id": int(row["customer_id"])}
        return obs, {}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset() before step() after episode is done.")
        action = int(action)
        row = self._get_customer_row()
        int_rate, unemp = self._macro_at(self.cur_month)

        # Apply action
        if action == 0:
            self.limit *= 0.9
        elif action == 2:
            self.limit *= 1.1
        # bounds
        self.limit = float(np.clip(self.limit, self.cfg.min_limit, self.cfg.max_limit))

        # Simulate spend and update balance (cap by limit*util_cap)
        spend = self._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], self.cur_month)
        spend = float(np.clip(spend, 0.0, self.limit * self.cfg.util_cap))

        # Add new spend to balance
        self.balance += spend

        # Draw repayment
        repay_rate = self._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp)
        payment = repay_rate * self.balance
        payment = float(np.clip(payment, 0.0, self.balance))
        self.balance -= payment

        util = np.clip(self.balance / max(self.limit, 1.0), 0.0, 1.0)
        def_prob = self._default_probability(util, repay_rate, unemp, int_rate, row["risk_score"])
        default_draw = self.rng.uniform()
        default_flag = default_draw < def_prob

        # Compute economics
        monthly_rate = self._monthly_rate(int_rate)
        interest = monthly_rate * self.balance
        fee = self._interchange(spend)
        profit = interest + fee

        reward = profit
        if default_flag:
            # Charge-off remaining balance
            loss = self.cfg.lgd * self.balance
            reward -= (loss + self.cfg.default_penalty)
            self.balance = 0.0  # account closed

        # NEW: per-step risk penalty (even if no default occurs)
        reward -= self.cfg.risk_penalty_lambda * def_prob

        # Overexposure penalty
        avg_spend = float(np.mean(self.spend_hist[-self.cfg.history_len:])) if self.spend_hist else spend
        reward -= self._overexposure_penalty(self.limit, avg_spend, def_prob)

        # Update histories
        self.spend_hist.append(spend)
        self.repay_hist.append(repay_rate)
        self.missed_hist.append(1 if repay_rate < 0.1 else 0)
        self.last_repay_rate = repay_rate

        self.cur_month += 1
        terminated = bool(default_flag or (self.cur_month >= self.T))
        truncated = False
        self.done = terminated

        obs = self._obs(int_rate, unemp)
        info = dict(
            customer_id=int(row["customer_id"]),
            spend=spend,
            payment=payment,
            repay_rate=repay_rate,
            interest=interest,
            fee=fee,
            profit=profit,
            default_prob=def_prob,
            default=default_flag,
            balance=self.balance,
            limit=self.limit,
            month=self.cur_month,
            macro_interest=int_rate,
            macro_unemployment=unemp,
            action=action,
        )
        self.info_last = info
        return obs, float(reward), terminated, truncated, info


# ---------- Batch simulator (shared by baseline + evaluation) ---------- #
def simulate_accounts(
    customers_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Optional[EnvConfig],
    policy_fn,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate all customers for config.months with a given policy_fn(state_dict)->action {0,1,2}.
    Returns monthly panel DataFrame with profitability and status.
    """
    cfg = config or EnvConfig()
    rng = np.random.default_rng(seed)
    rows = []

    env = CreditLimitEnv(customers_df, macro_df, cfg)

    for i in range(len(customers_df)):
        # Force specific customer for repeatability
        env.customer_idx = i
        env.cur_month = 0
        env.done = False
        # Reset-like init for fixed customer
        row = env._get_customer_row()
        env.limit = float(np.clip(row["init_limit"], cfg.min_limit, cfg.max_limit))
        env.balance = 0.0
        env.last_repay_rate = 1.0
        env.spend_hist, env.repay_hist, env.missed_hist = [], [], []
        int0, unemp0 = env._macro_at(0)
        for _ in range(cfg.history_len):
            spend0 = env._draw_spend(row["base_spend"], row["spend_std"], row["pattern"], 0)
            rr0 = env._draw_repay_rate(row["repay_behavior"], row["base_repay_mean"], row["base_repay_std"], unemp0)
            env.spend_hist.append(spend0)
            env.repay_hist.append(rr0)
            env.missed_hist.append(1 if rr0 < 0.1 else 0)

        for _ in range(env.T):
            if env.done:
                break
            int_rate, unemp = env._macro_at(env.cur_month)
            state_dict = {
                "limit_norm": env.limit / cfg.limit_scale,
                "avg_spend_norm": (np.mean(env.spend_hist[-cfg.history_len:]) / cfg.limit_scale) if env.spend_hist else 0.0,
                "repay_rate": env.last_repay_rate,
                "util": float(np.clip(env.balance / max(env.limit, 1.0), 0.0, 1.0)),
                "interest_rate": float(int_rate),
                "unemployment_rate": float(unemp),
            }
            action = int(policy_fn(state_dict))

            # Apply action
            if action == 0:
                env.limit *= 0.9
            elif action == 2:
                env.limit *= 1.1
            env.limit = float(np.clip(env.limit, cfg.min_limit, cfg.max_limit))

            # Spend + balance dynamics
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

            # NEW: per-step risk penalty mirrors env.step()
            reward -= cfg.risk_penalty_lambda * def_prob

            avg_spend = float(np.mean(env.spend_hist[-cfg.history_len:])) if env.spend_hist else spend
            reward -= env._overexposure_penalty(env.limit, avg_spend, def_prob)

            env.spend_hist.append(spend)
            env.repay_hist.append(repay_rate)
            env.missed_hist.append(1 if repay_rate < 0.1 else 0)
            env.last_repay_rate = repay_rate

            env.cur_month += 1
            terminated = bool(default_flag or (env.cur_month >= env.T))
            env.done = terminated

            rows.append(dict(
                customer_id=int(row["customer_id"]),
                month=int(env.cur_month),
                action=action,
                limit=float(env.limit),
                spend=float(spend),
                payment=float(payment),
                repay_rate=float(repay_rate),
                interest=float(interest),
                fee=float(fee),
                profit=float(profit),
                reward=float(reward),
                default_prob=float(def_prob),
                default=int(default_flag),
                balance=float(env.balance),
                macro_interest=float(int_rate),
                macro_unemployment=float(unemp),
            ))

    return pd.DataFrame(rows)
