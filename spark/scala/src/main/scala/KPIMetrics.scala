package com.acme
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object KPIMetrics {
  def main(args: Array[String]): Unit = {
    val db = if (args.nonEmpty) args(0) else "credit_rl"
    val spark = SparkSession.builder.appName("CreditRL-KPIs").enableHiveSupport().getOrCreate()
    val panel = spark.table(s"$db.simulation_panel") // write your simulate outputs here if you persist to Hive
    val byMonth = panel.groupBy("month")
      .agg(sum("profit").as("profit_m"), sum("reward").as("reward_m"),
           sum(col("default").cast("int")).as("defaults_m"), avg("limit").as("avg_limit_m"))
      .orderBy("month")
    byMonth.write.mode("overwrite").format("parquet").saveAsTable(s"$db.kpi_monthly")
    spark.stop()
  }
}
