from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
from pyspark.sql.functions import col

custom_temp_dir = "/media/dori/Data SSD/cihuy"

# Initialize Spark session with increased memory
spark = SparkSession.builder \
    .appName("AnimeMultipleModelsTraining") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "16g") \
    .config("spark.yarn.executor.memoryOverhead", "8g") \
    .getOrCreate()

# Load data for each model from specific JSON files
model_1_data = spark.read.json('batches/batch_1_1731403729.json')
model_2_data = spark.read.json('batches/batch_2_1731403772.json')
model_3_data = spark.read.json('batches/batch_3_1731403810.json')

# Combine data into a single DataFrame if needed
all_data = model_1_data.union(model_2_data).union(model_3_data).repartition(10000)

# Check the schema of the DataFrame
all_data.printSchema()

# Convert columns to numeric types if they are not already
for col_name in ["popularity", "rank", "scored_by", "my_score"]:
    all_data = all_data.withColumn(col_name, col(col_name).cast("float"))

# Drop rows with null values that may have resulted from conversion
all_data = all_data.na.drop()

# Define features and target
features = ["popularity", "rank", "scored_by"]
target = "my_score"

# Prepare lists to store models and evaluators
models = {"LinearRegression": [], "DecisionTree": [], "GradientBoostedTree": []}
evaluators = {"LinearRegression": [], "DecisionTree": [], "GradientBoostedTree": []}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Data Preprocessing
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(all_data).select("features", target).cache()  # Cache the data

train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Linear Regression Model
lr = LinearRegression(labelCol=target, featuresCol="features")
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)
lr_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
lr_rmse = lr_evaluator.evaluate(lr_predictions)
models["LinearRegression"].append(lr_model)
evaluators["LinearRegression"].append(lr_rmse)
lr_model.save('models/lr_model')

# Decision Tree Regressor Model
dt = DecisionTreeRegressor(labelCol=target, featuresCol="features")
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)
dt_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
dt_rmse = dt_evaluator.evaluate(dt_predictions)
models["DecisionTree"].append(dt_model)
evaluators["DecisionTree"].append(dt_rmse)
dt_model.save('models/dt_model')

# Gradient-Boosted Tree Regressor Model
gbt = GBTRegressor(labelCol=target, featuresCol="features")
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)
gbt_evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
models["GradientBoostedTree"].append(gbt_model)
evaluators["GradientBoostedTree"].append(gbt_rmse)
gbt_model.save('models/gbt_model')

# Print results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Decision Tree RMSE: {dt_rmse}")
print(f"Gradient-Boosted Tree RMSE: {gbt_rmse}")

spark.stop()
