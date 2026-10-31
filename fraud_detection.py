"""
INT315 - Cluster Computing Project
Distributed Credit Card Fraud Detection using Apache Spark MLlib
Run from CMD:  python fraud_detection.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ---------------------------------------------------------------
# 0. Start Spark session (local mode, uses all CPU cores)
# ---------------------------------------------------------------
spark = (SparkSession.builder
         .appName("FraudDetection_INT315")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("\nSpark version :", spark.version)
print("Spark UI      : http://localhost:4040")
print("Default parallelism (cores):", spark.sparkContext.defaultParallelism)

# ---------------------------------------------------------------
# 1. Load data and basic exploration
# ---------------------------------------------------------------
print("\n===== 1. LOAD DATA & EXPLORE =====")
df = spark.read.csv("creditcard.csv", header=True, inferSchema=True)
df.printSchema()
total_rows = df.count()
print("Rows:", total_rows, "| Columns:", len(df.columns))
print("Rows containing nulls:", total_rows - df.dropna().count())

print("Class distribution (0 = normal, 1 = fraud):")
df.groupBy("Class").count().show()
df.select("Time", "Amount").describe().show()

# ---------------------------------------------------------------
# 2. Handle class imbalance using class weights
# ---------------------------------------------------------------
print("\n===== 2. CLASS WEIGHTS =====")
n_neg = df.filter(F.col("Class") == 0).count()
n_pos = df.filter(F.col("Class") == 1).count()
ratio = n_neg / n_pos
print(f"Normal: {n_neg} | Fraud: {n_pos} | Fraud weight: {ratio:.2f}")
df = df.withColumn("weight", F.when(F.col("Class") == 1, ratio).otherwise(1.0))

# ---------------------------------------------------------------
# 3. Train / test split (80 / 20)
# ---------------------------------------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)
train.cache()
test.cache()
print("\nTrain rows:", train.count(), "| Test rows:", test.count())

# ---------------------------------------------------------------
# 4. Feature pipeline stages
# ---------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ("Class", "weight")]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(labelCol="Class", featuresCol="features",
                        weightCol="weight", maxIter=20)
rf = RandomForestClassifier(labelCol="Class", featuresCol="features",
                            weightCol="weight", numTrees=50, maxDepth=8, seed=42)


# ---------------------------------------------------------------
# 5. Evaluation helper
# ---------------------------------------------------------------
def evaluate(name, pred):
    auc_roc = BinaryClassificationEvaluator(
        labelCol="Class", metricName="areaUnderROC").evaluate(pred)
    auc_pr = BinaryClassificationEvaluator(
        labelCol="Class", metricName="areaUnderPR").evaluate(pred)

    tp = pred.filter((F.col("Class") == 1) & (F.col("prediction") == 1)).count()
    tn = pred.filter((F.col("Class") == 0) & (F.col("prediction") == 0)).count()
    fp = pred.filter((F.col("Class") == 0) & (F.col("prediction") == 1)).count()
    fn = pred.filter((F.col("Class") == 1) & (F.col("prediction") == 0)).count()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"\n--- {name} ---")
    print("Confusion Matrix (rows = actual, cols = predicted)")
    print("              Pred 0    Pred 1")
    print(f"Actual 0    {tn:8d}  {fp:8d}")
    print(f"Actual 1    {fn:8d}  {tp:8d}")
    return {"Model": name, "AUC-ROC": auc_roc, "AUC-PR": auc_pr,
            "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ---------------------------------------------------------------
# 6. Train and evaluate both models
# ---------------------------------------------------------------
print("\n===== 3. MODEL TRAINING & EVALUATION =====")
results = []
for name, algo in [("Logistic Regression", lr), ("Random Forest", rf)]:
    start = time.time()
    model = Pipeline(stages=[assembler, scaler, algo]).fit(train)
    train_time = time.time() - start
    pred = model.transform(test)
    res = evaluate(name, pred)
    res["Train time (s)"] = train_time
    results.append(res)

print("\n===== 4. MODEL COMPARISON =====")
header = f"{'Model':22s}{'AUC-ROC':>9s}{'AUC-PR':>9s}{'Acc':>8s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}{'Time(s)':>9s}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['Model']:22s}{r['AUC-ROC']:9.4f}{r['AUC-PR']:9.4f}{r['Accuracy']:8.4f}"
          f"{r['Precision']:8.4f}{r['Recall']:8.4f}{r['F1']:8.4f}{r['Train time (s)']:9.2f}")

# ---------------------------------------------------------------
# 7. Partition experiment (cluster computing angle)
# ---------------------------------------------------------------
print("\n===== 5. PARTITION EXPERIMENT (Logistic Regression) =====")
print(f"{'Partitions':>10s}{'Train time (s)':>18s}")
part_times = {}
for n in [2, 4, 8]:
    part_train = train.repartition(n)
    part_train.count()  # materialize before timing
    start = time.time()
    Pipeline(stages=[assembler, scaler, lr]).fit(part_train)
    part_times[n] = time.time() - start
    print(f"{n:10d}{part_times[n]:18.2f}")

# ---------------------------------------------------------------
# 8. Visualisations (saved as PNG files in the "charts" folder)
# ---------------------------------------------------------------
print("\n===== 6. VISUALISATIONS =====")
import os
import matplotlib
matplotlib.use("Agg")  # no GUI needed, works in CMD
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)

# Chart 1: class distribution (log scale because fraud is very rare)
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["Normal (0)", "Fraud (1)"], [n_neg, n_pos], color=["steelblue", "crimson"])
ax.set_yscale("log")
ax.set_ylabel("Number of transactions (log scale)")
ax.set_title("Class Distribution")
for b, v in zip(bars, [n_neg, n_pos]):
    ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
fig.tight_layout()
fig.savefig("charts/1_class_distribution.png", dpi=150)
plt.close(fig)

# Chart 2: model comparison
metrics = ["AUC-ROC", "AUC-PR", "Precision", "Recall", "F1"]
fig, ax = plt.subplots(figsize=(8, 4))
width = 0.35
for i, r in enumerate(results):
    xs = [j + i * width for j in range(len(metrics))]
    ax.bar(xs, [r[m] for m in metrics], width, label=r["Model"])
ax.set_xticks([j + width / 2 for j in range(len(metrics))])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_title("Model Comparison")
ax.legend()
fig.tight_layout()
fig.savefig("charts/2_model_comparison.png", dpi=150)
plt.close(fig)

# Chart 3: confusion matrices
fig, axes = plt.subplots(1, len(results), figsize=(9, 4))
for ax, r in zip(axes, results):
    cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(r["Model"])
fig.tight_layout()
fig.savefig("charts/3_confusion_matrices.png", dpi=150)
plt.close(fig)

# Chart 4: partitions vs training time
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar([str(k) for k in part_times], list(part_times.values()), color="seagreen")
ax.set_xlabel("Number of partitions")
ax.set_ylabel("Training time (s)")
ax.set_title("Effect of Partitions on Training Time")
fig.tight_layout()
fig.savefig("charts/4_partition_time.png", dpi=150)
plt.close(fig)

print("Charts saved in:", os.path.abspath("charts"))

print("\nDone. Spark UI is live at http://localhost:4040 - take screenshots now.")
input("Press ENTER to close Spark and exit...")
spark.stop()
