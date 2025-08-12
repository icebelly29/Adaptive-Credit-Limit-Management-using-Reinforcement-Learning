from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_from_parquet(parquet_dir: str | Path):
    p = Path(parquet_dir)
    customers = pd.read_parquet(p / "customers")
    macro = pd.read_parquet(p / "macro")
    return customers, macro

def load_from_hive(db: str):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.enableHiveSupport().appName("CreditRL-Load").getOrCreate())
    customers = spark.table(f"{db}.customers").toPandas()
    macro = spark.table(f"{db}.macro").toPandas()
    spark.stop()
    return customers, macro
