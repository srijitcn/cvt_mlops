# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``cvt_mlops_stacks_demo/resources/model-workflow-resource.yml``

#  
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["dev","staging", "prod"], "Environment Name")

# user_uuid
dbutils.widgets.text("user_uuid","8243",label="User UUID")

# MLflow experiment name.
dbutils.widgets.text("experiment_name","/mlops_demo/dev_cvt_tumor_classifier",label="MLflow experiment name")

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text("model_name", "mlops_demo.cvt.cvt_tumor_classifier", label="Full (Three-Level) Model Name")

# Feature table to store the computed features.
dbutils.widgets.text("patch_feature_table_name","mlops_demo.cvt.patch_features",label="Output Feature Table Name")

# Primary Keys columns for the feature table;
dbutils.widgets.text("patch_primary_keys", "id",label="Primary keys columns for the patch feature table, comma separated.")

# COMMAND ----------

# DBTITLE 1,Define input and output variables
env = dbutils.widgets.get("env")
user_uid = dbutils.widgets.get("user_uuid")
feature_table = dbutils.widgets.get("patch_feature_table_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Helper functions
from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import torch
import torchvision
from PIL import Image
from functools import partial 
from petastorm import TransformSpec
from torchvision import transforms

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.torch as hvd
from sparkdl import HorovodRunner

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# COMMAND ----------

import json
import os
from pprint import pprint

project_name='digital-pathology'

config_path=f"/dbfs/FileStore/mlops_digital_pathology/{env}/{user_uid}_{project_name}_configs.json"

try:
  with open(config_path,'rb') as f:
    settings = json.load(f)
except FileNotFoundError:
  print('please run ./config notebook and try again')
  assert False

# COMMAND ----------

IMG_PATH = settings['img_path']
experiment_info=mlflow.set_experiment(settings['experiment_name'])

# COMMAND ----------

# DBTITLE 1, Create FeatureLookups
from databricks.feature_store import FeatureLookup
import mlflow

feature_lookups = [
    FeatureLookup(
        table_name=feature_table,
        feature_names=[
            "features"
        ],
        lookup_key=["id"]
    ),
]


# COMMAND ----------

# DBTITLE 1, Create Training Dataset
from databricks import feature_store

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()


fs = feature_store.FeatureStoreClient()

#this can be an input data table from gold/silver. For this example we are using feature table
input_data = spark.table(feature_table).select("id","label").limit(20)

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
    input_data,
    feature_lookups=feature_lookups,
    label="label",
    exclude_columns=["content","id"],
)

# Load the TrainingSet into a dataframe
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, 
training_df.display()

# COMMAND ----------

df_train, df_val = training_df.randomSplit([0.7, 0.3], seed=12345)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Train Model

# COMMAND ----------

from pyspark import keyword_only
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame

from custom_transformers import DenseVectorizer

# used as a multi class classifier
lr = LogisticRegression(maxIter=1, regParam=0.03, elasticNetParam=0.5, labelCol="label", featuresCol="feature")
dv = DenseVectorizer(inputCol="features",outputCol="dense_features")
va = VectorAssembler(inputCols=["dense_features"],outputCol="feature")

# define a pipeline model
model = Pipeline(stages=[dv,va,lr])

spark_model = model.fit(df_train) # start fitting or training

# COMMAND ----------

df_val.count()

# COMMAND ----------

pred_df = spark_model.transform(df_val)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator() 

f1_score = evaluator.evaluate(pred_df, {evaluator.metricName: 'f1'})
precision = evaluator.evaluate(pred_df,{evaluator.metricName: 'weightedPrecision'})
recall = evaluator.evaluate(pred_df, {evaluator.metricName: 'weightedRecall'})
accuracy = evaluator.evaluate(pred_df, {evaluator.metricName: 'accuracy'})

print('F1-Score ', f1_score)
print('Precision ', precision)
print('Recall ', recall)
print('Accuracy ', accuracy)

# COMMAND ----------

# DBTITLE 1, Log model and return output.

# Log the trained model with MLflow and package it with feature lookup information.
fs.log_model(
    spark_model,
    artifact_path="model_packaged",
    flavor=mlflow.spark,
    training_set=training_set,
    registered_model_name=model_name, 
    code_paths=["./custom_transformers.py"]
)


# COMMAND ----------

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
#dbutils.notebook.exit(model_uri)

# COMMAND ----------

model_uri

# COMMAND ----------

display(input_data)

# COMMAND ----------

res_df = fs.score_batch(model_uri,input_data)

# COMMAND ----------

display(res_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trying to do Non Feature Store scoring

# COMMAND ----------



# COMMAND ----------

mm = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

display(training_df.withColumn("preds", mm("features")))

# COMMAND ----------


