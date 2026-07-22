from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

HDFS_INPUT  = "hdfs://localhost:9000/progetto_bigdata/CDs_and_Vinyl.jsonl"
HDFS_OUTPUT = "hdfs://localhost:9000/progetto_bigdata/dataset_grezzo"
LOCAL_OUTPUT = "dataset_grezzo.csv"
SAMPLE_FRACTION = 0.1
SEED = 42


def main():
    print("=" * 60)
    print("STEP 01 — Data Ingestion & Sampling")
    print("=" * 60)

    spark = SparkSession.builder \
        .appName("Step01_Ingestion") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    #1. Lettura da HDFS 
    print("\n[1/5] Lettura dataset da HDFS...")
    df_raw = spark.read.json(HDFS_INPUT)
    total_raw = df_raw.count()
    print(f"      Righe totali raw: {total_raw:,}")

   
    df = df_raw.select(
        F.col("text"),                                
        F.col("rating").cast("float"),                
        F.col("asin"),
        F.col("user_id").alias("reviewerID")          
    )

    #3. Filtri qualitativi 
    print("\n[2/5] Applicazione filtri qualitativi...")
    df_clean = df \
        .filter(F.col("text").isNotNull()) \
        .filter(F.trim(F.col("text")) != "") \
        .filter(F.length(F.col("text")) >= 10) \
        .filter(F.col("rating").isNotNull()) \
        .filter(F.col("rating").isin([1.0, 2.0, 3.0, 4.0, 5.0]))

    total_clean = df_clean.count()
    print(f"      Righe dopo pulizia: {total_clean:,} ({100*total_clean/total_raw:.1f}% del raw)")

    # 4. Analisi distribuzione classi (pre-campionamento) 
    print("\n[3/5] Distribuzione classi nel dataset pulito:")
    df_clean.groupBy("rating").count().orderBy("rating").show()

    # 5. Campionamento stratificato 10% 
    print(f"[4/5] Campionamento stratificato {SAMPLE_FRACTION*100:.0f}% (seed={SEED})...")
    fractions = {r: SAMPLE_FRACTION for r in [1.0, 2.0, 3.0, 4.0, 5.0]}
    df_sample = df_clean.sampleBy("rating", fractions=fractions, seed=SEED)

    total_sample = df_sample.count()
    print(f"      Righe campionate: {total_sample:,}")
    df_sample.groupBy("rating").count().orderBy("rating").show()

    # 6. Salvataggio 
    print("[5/5] Salvataggio dataset grezzo campionato...")
    df_pandas = df_sample.select("text", "rating", "asin", "reviewerID").toPandas()
    df_pandas.to_csv(LOCAL_OUTPUT, index=False)
    print(f"       Salvato: '{LOCAL_OUTPUT}' ({len(df_pandas):,} righe)")

    spark.stop()


if __name__ == "__main__":
    main()
