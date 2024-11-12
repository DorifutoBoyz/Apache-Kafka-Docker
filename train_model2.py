from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
from pyspark.sql.functions import col

# Initialize Spark session with increased memory
spark = SparkSession.builder \
    .appName("AnimeMultipleModelsTraining") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.offHeap.enabled","true") \
    .config("spark.memory.offHeap.size","16g") \
    .config("spark.yarn.executor.memoryOverhead", "8g") \
    .getOrCreate()

# Load all data from the directory

all_data = spark.read.json('batches/*.json').repartition(10000)  # Adjust the number of partitions


# Check the schema of the DataFrame

all_data.printSchema()


# Convert columns to numeric types if they are not already

all_data = all_data.withColumn("popularity", col("popularity").cast("float"))

all_data = all_data.withColumn("rank", col("rank").cast("float"))

all_data = all_data.withColumn("scored_by", col("scored_by").cast("float"))

all_data = all_data.withColumn("my_score", col("my_score").cast("float"))  # Convert target variable


# Drop rows with null values that may have resulted from conversion

all_data = all_data.na.drop()


# Define features and target

features = ["popularity", "rank", "scored_by"]

target = "my_score"

# Get the total number of records
total_records = all_data.count()
batch_size = 500000

# Prepare lists to store models and evaluators
models = {"LinearRegression": [], "DecisionTree": [], "GradientBoostedTree": []}
evaluators = {"LinearRegression": [], "DecisionTree": [], "GradientBoostedTree": []}

# Process the data in chunks of 500,000
for i in range(0, total_records, batch_size):
    batch_data = all_data.limit(batch_size).offset(i)  # Adjust this line to select the correct batch

    # Data Preprocessing
    batch_data = batch_data.na.drop()
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(batch_data).select("features", target).cache()  # Cache the data

    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

    # Linear Regression Model
    lr = LinearRegression(labelCol=target, featuresCol="features")
    lr_model = lr.fit(train_data)
    lr_predictions = lr_model.transform(test_data)
    lr_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    lr_rmse = lr_evaluator.evaluate(lr_predictions)
    models["LinearRegression"].append(lr_model)
    evaluators["LinearRegression"].append(lr_rmse)
    lr_model.save(f'models/lr_model_batch_{i//batch_size + 1}')

    # Decision Tree Regressor Model
    dt = DecisionTreeRegressor(labelCol=target, featuresCol="features")
    dt_model = dt.fit(train_data)
    dt_predictions = dt_model.transform(test_data)
    dt_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    dt_rmse = dt_evaluator.evaluate(dt_predictions)
    models["DecisionTree"].append(dt_model)
    evaluators["DecisionTree"].append(dt_rmse)
    dt_model.save(f'models/dt_model_batch_{i//batch_size + 1}')

    # Gradient-Boosted Tree Regressor Model
    gbt = GBTRegressor(labelCol=target, featuresCol="features")
    gbt_model = gbt.fit(train_data)
    gbt_predictions = gbt_model.transform(test_data)
    gbt_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
    models["GradientBoostedTree"].append(gbt_model)
    evaluators["GradientBoostedTree"].append(gbt_rmse)
    gbt_model.save(f'models/gbt_model_batch_{i//batch_size + 1}')

    print(f"Batch {i//batch_size + 1} Results:")
    print(f"Linear Regression RMSE: {lr_rmse}")
    print(f"Decision Tree RMSE: {dt_rmse}")
    print(f"Gradient-Boosted Tree RMSE: {gbt_rmse}")

spark.stop()
