# Adaptive Credit Limit Management using Reinforcement Learning

This project trains a PPO agent (Stable-Baselines3) in a custom Gymnasium environment to **dynamically adjust credit card limits** based on:
- Customer features (utilization, spend, repayment behavior)
- Macroeconomic indicators (interest rate, unemployment)

It includes a **static baseline policy** for comparison, **synthetic data generation**, robust **evaluation & plotting**, and optional **stress tests** (recession shock) and **explainability hooks**.


### optional: clear caches
```powershell
rmdir /s /q __pycache__ 2> NUL
for /d %D in (*) do if exist "%D\__pycache__" rmdir /s /q "%D\__pycache__"
```

## How to run
```bash
python -m pip install -r requirements.txt
python baseline_policy.py
python train_rl_agent.py
python evaluate.py --model outputs\models\ppo_credit_limit.zip --customers 1000 --months 60 --seed 7 --shock
```