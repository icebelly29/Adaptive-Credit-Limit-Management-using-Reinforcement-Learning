# Run: spark-submit --master local[*] spark/pyspark_generate.py --customers 1000000 --months 60 --db credit_rl
import argparse, numpy as np
from pyspark.sql import SparkSession, functions as F

p = argparse.ArgumentParser()
p.add_argument("--customers", type=int, default=1_000_000)
p.add_argument("--months", type=int, default=60)
p.add_argument("--db", type=str, default="credit_rl")
args = p.parse_args()

spark = (SparkSession.builder.appName("GenerateCreditRLData").enableHiveSupport().getOrCreate())
spark.sql(f"CREATE DATABASE IF NOT EXISTS {args.db}")

n = args.customers
rnd = F.rand(seed=7)
base = spark.range(0, n).withColumnRenamed("id","customer_id")

pattern = F.when(rnd<0.55,"normal").when(rnd<0.80,"high").otherwise("seasonal")
behavior= F.when(rnd<0.55,"on_time").when(rnd<0.80,"late").otherwise("partial")

base_spend = F.when(pattern=="normal", F.expr("500 + rand()*700")).when(pattern=="high", F.expr("1200 + rand()*1300")).otherwise(F.expr("600 + rand()*900"))
spend_std = F.when(pattern=="normal", F.expr("80 + rand()*120")).when(pattern=="high", F.expr("150 + rand()*200")).otherwise(F.expr("100 + rand()*150"))
base_repay_mean = F.when(behavior=="on_time", F.expr("0.85 + rand()*0.13")).when(behavior=="late", F.expr("0.55 + rand()*0.25")).otherwise(F.expr("0.25 + rand()*0.30"))
base_repay_std  = F.when(behavior=="on_time", F.expr("0.02 + rand()*0.05")).when(behavior=="late", F.expr("0.05 + rand()*0.07")).otherwise(F.expr("0.07 + rand()*0.08"))
risk_score = F.rand(seed=101)

customers = (base
  .withColumn("pattern", pattern)
  .withColumn("repay_behavior", behavior)
  .withColumn("base_spend", base_spend)
  .withColumn("spend_std", spend_std)
  .withColumn("base_repay_mean", base_repay_mean)
  .withColumn("base_repay_std", base_repay_std)
  .withColumn("risk_score", risk_score)
  .withColumn("init_limit", F.expr("LEAST(20000.0, GREATEST(500.0, (base_spend*(2.5 + rand()*1.5))*(1.2 - 0.8*risk_score)))"))
)
customers.write.mode("overwrite").format("parquet").saveAsTable(f"{args.db}.customers")

# Macro random walk with a shock at t=24
T = args.months
rates = [0.05]; unemp = [0.05]
rng = np.random.default_rng(108)
for t in range(1, T):
    r = rates[-1] + rng.normal(0,0.002) - 0.05*(rates[-1]-0.05)
    u = unemp[-1] + rng.normal(0,0.003) - 0.1*(unemp[-1]-0.05)
    rates.append(float(max(0.0, min(0.25, r)))); unemp.append(float(max(0.02, min(0.20, u))))
if T>24:
    rates[24] = min(0.30, rates[24] + 0.02)
    unemp[24] = min(0.35, unemp[24] + 0.03)
macro = spark.createDataFrame(list(zip(range(T), rates, unemp)), "month INT, interest_rate DOUBLE, unemployment_rate DOUBLE")
macro.write.mode("overwrite").format("parquet").saveAsTable(f"{args.db}.macro")

print(f"Wrote tables: {args.db}.customers, {args.db}.macro")
spark.stop()
