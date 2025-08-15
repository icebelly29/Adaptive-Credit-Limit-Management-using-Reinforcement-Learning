# Adaptive Credit Limit Management using Reinforcement Learning


End-to-end project with:
- **RL (Python, PyTorch, SB3, Gymnasium)** for credit limit adjustment
- **Big Data (PySpark + Hive)** to generate/hold large synthetic cohorts
- **Scala Spark** KPI job (optional)
- **Java (DJL + TorchScript)** to serve the trained policy


## How to run

### optional: clear caches
```cmd
rmdir /s /q __pycache__ 2> NUL
for /d %D in (*) do if exist "%D\__pycache__" rmdir /s /q "%D\__pycache__"
```
### run:
```cmd
python -m pip install -r requirements.txt
python baseline_policy.py
python train_rl_agent.py
python evaluate.py --model outputs\models\ppo_credit_limit.zip --customers 1000 --months 60 --seed 7 --shock
```