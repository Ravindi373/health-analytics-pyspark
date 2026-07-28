"""
Student Health & Biodata Analytics with PySpark
=================================================

A small end-to-end PySpark project that loads a student biodata dataset,
cleans it, engineers health-risk features (BMI, hypertension flag, at-risk
flag), and produces a set of aggregated insights.

Run locally (see README.md for setup):
    python src/health_analytics.py
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DATA_PATH = "data/biodata_advanced.csv"
OUTPUT_PATH = "outputs/at_risk_students.csv"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("StudentHealthAnalytics")
        .master("local[*]")
        .getOrCreate()
    )


def load_data(spark: SparkSession):
    print("\n=== PART A: DATA LOADING ===")
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

    print("\nFirst 5 rows:")
    df.show(5, truncate=False)

    print(f"Dataset shape: ({df.count()} rows, {len(df.columns)} columns)")

    print("\nSchema / data types:")
    df.printSchema()

    return df


def explore_data(df):
    print("\n=== PART B: DATA CLEANING & PREPROCESSING ===")

    print("\nSummary statistics (describe):")
    df.describe().show()

    print("\nMissing / null value counts per column:")
    df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show()

    print("Zero-value counts in fields where zero is biologically invalid:")
    for c in ["height_cm", "weight_kg", "cholesterol_mg_dl"]:
        zero_count = df.filter(F.col(c) == 0).count()
        print(f"  {c}: {zero_count} rows with value 0")

    return df


def clean_data(df):
    # Step 1: treat 0 in height/weight/cholesterol as missing (invalid),
    # since a living person cannot have zero height, weight, or cholesterol.
    for c in ["height_cm", "weight_kg", "cholesterol_mg_dl"]:
        df = df.withColumn(c, F.when(F.col(c) == 0, None).otherwise(F.col(c)))

    # Step 2 (Option B - imputation): fill missing numeric values with the
    # mean for that (gender, department) group. This preserves sample size
    # (important with only 100 rows) instead of dropping rows outright, and
    # keeps values realistic since height/weight/cholesterol vary by gender
    # and, to a lesser extent, by department/lifestyle.
    numeric_cols = ["height_cm", "weight_kg", "cholesterol_mg_dl"]
    group_window = Window.partitionBy("gender", "department")

    for c in numeric_cols:
        df = df.withColumn(
            f"{c}_group_mean", F.avg(F.col(c)).over(group_window)
        )
        df = df.withColumn(
            c, F.coalesce(F.col(c), F.round(F.col(f"{c}_group_mean"), 1))
        ).drop(f"{c}_group_mean")

    # Fallback: if an entire (gender, department) group had all values
    # missing for a column, fill any remaining nulls with the global mean.
    for c in numeric_cols:
        global_mean = df.select(F.round(F.avg(c), 1)).first()[0]
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(global_mean)))

    print("\nMissing values remaining after cleaning:")
    df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).show()

    return df


def engineer_features(df):
    print("\n=== PART C: FEATURE ENGINEERING ===")

    df = df.withColumn(
        "bmi", F.round(F.col("weight_kg") / F.pow(F.col("height_cm") / 100, 2), 2)
    )

    df = df.withColumn(
        "bmi_category",
        F.when(F.col("bmi") < 18.5, "Underweight")
        .when(F.col("bmi") < 25, "Normal")
        .when(F.col("bmi") < 30, "Overweight")
        .otherwise("Obese"),
    )

    df = df.withColumn(
        "high_bp",
        (F.col("systolic_bp") >= 140) | (F.col("diastolic_bp") >= 90),
    )

    df = df.withColumn(
        "at_risk",
        F.col("bmi_category").isin("Overweight", "Obese")
        | F.col("high_bp")
        | (F.col("cholesterol_mg_dl") >= 200),
    )

    print("\nSample of engineered columns:")
    df.select(
        "student_id", "name", "bmi", "bmi_category", "high_bp", "at_risk"
    ).show(10, truncate=False)

    return df


def analyze(df):
    print("\n=== PART D: ANALYSIS & AGGREGATION ===")

    print("\nAverage BMI per department:")
    avg_bmi_dept = (
        df.groupBy("department")
        .agg(F.round(F.avg("bmi"), 2).alias("avg_bmi"))
        .orderBy(F.desc("avg_bmi"))
    )
    avg_bmi_dept.show()

    print("Percentage of at-risk students per department:")
    at_risk_pct = (
        df.groupBy("department")
        .agg(
            F.round(100 * F.avg(F.col("at_risk").cast("int")), 1).alias(
                "at_risk_pct"
            ),
            F.count("*").alias("num_students"),
        )
        .orderBy(F.desc("at_risk_pct"))
    )
    at_risk_pct.show()

    print("Mean systolic / diastolic BP by year:")
    bp_by_year = (
        df.groupBy("year")
        .agg(
            F.round(F.avg("systolic_bp"), 1).alias("mean_systolic"),
            F.round(F.avg("diastolic_bp"), 1).alias("mean_diastolic"),
        )
        .orderBy("year")
    )
    bp_by_year.show()

    print("Pivot table: department x gender -> mean BMI:")
    pivot = (
        df.groupBy("department")
        .pivot("gender")
        .agg(F.round(F.avg("bmi"), 2))
        .orderBy("department")
    )
    pivot.show()

    print("At-risk students (sorted by BMI, descending):")
    at_risk_students = (
        df.filter(F.col("at_risk"))
        .select(
            "student_id",
            "name",
            "department",
            "bmi",
            "bmi_category",
            "systolic_bp",
            "diastolic_bp",
            "cholesterol_mg_dl",
            "high_bp",
        )
        .orderBy(F.desc("bmi"))
    )
    at_risk_students.show(at_risk_students.count(), truncate=False)

    return avg_bmi_dept, at_risk_pct, bp_by_year, pivot, at_risk_students


def export_results(at_risk_students):
    print("\n=== PART E: EXPORTING RESULTS ===")
    (
        at_risk_students.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv("outputs/_at_risk_students_tmp")
    )

    import glob
    import shutil

    part_file = glob.glob("outputs/_at_risk_students_tmp/part-*.csv")[0]
    shutil.move(part_file, OUTPUT_PATH)
    shutil.rmtree("outputs/_at_risk_students_tmp")
    print(f"Exported at-risk students to: {OUTPUT_PATH}")


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_data(spark)
    explore_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    _, _, _, _, at_risk_students = analyze(df)
    export_results(at_risk_students)

    print("\nDone. See outputs/at_risk_students.csv and README.md for findings.")
    spark.stop()


if __name__ == "__main__":
    main()
