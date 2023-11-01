# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

##################################################################################
# Model Validation Notebook
##
# This notebook uses mlflow model validation API to run mode validation after training and registering a model
# in model registry, before deploying it to the "Champion" alias.
#
# It runs as part of CD and by an automated model training job -> validation -> deployment job defined under ``cvt_mlops_stacks_demo/resources/model-workflow-resource.yml``
#
#
# Parameters:
#
# * env                                     - Name of the environment the notebook is run in (staging, or prod). Defaults to "prod".
# * `run_mode`                              - The `run_mode` defines whether model validation is enabled or not. It can be one of the three values:
#                                             * `disabled` : Do not run the model validation notebook.
#                                             * `dry_run`  : Run the model validation notebook. Ignore failed model validation rules and proceed to move
#                                                            model to the "Champion" alias.
#                                             * `enabled`  : Run the model validation notebook. Move model to the "Champion" alias only if all model validation
#                                                            rules are passing.
# * enable_baseline_comparison              - Whether to load the current registered "Champion" model as baseline.
#                                             Baseline model is a requirement for relative change and absolute change validation thresholds.
# * validation_input                        - Validation input. Please refer to data parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * model_type                              - A string describing the model type. The model type can be either "regressor" and "classifier".
#                                             Please refer to model_type parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * targets                                 - The string name of a column from data that contains evaluation labels.
#                                             Please refer to targets parameter in mlflow.evaluate documentation https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# * custom_metrics_loader_function          - Specifies the name of the function in cvt_mlops_stacks_demo/validation/validation.py that returns custom metrics.
# * validation_thresholds_loader_function   - Specifies the name of the function in cvt_mlops_stacks_demo/validation/validation.py that returns model validation thresholds.
#
# For details on mlflow evaluate API, see doc https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
# For details and examples about performing model validation, see the Model Validation documentation https://mlflow.org/docs/latest/models.html#model-validation
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

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../

# COMMAND ----------

dbutils.widgets.text("experiment_name","/mlops_demo/dev_cvt_tumor_classifier","Experiment Name")
dbutils.widgets.dropdown("run_mode", "enabled", ["disabled", "dry_run", "enabled"], "Run Mode")
dbutils.widgets.dropdown("enable_baseline_comparison", "false", ["true", "false"], "Enable Baseline Comparison")

dbutils.widgets.text("model_type", "classifier", "Model Type")
dbutils.widgets.text("targets", "label", "Targets")
dbutils.widgets.text("custom_metrics_loader_function", "custom_metrics", "Custom Metrics Loader Function")
dbutils.widgets.text("validation_input", "SELECT id,features,label FROM mlops_demo.cvt.patch_features LIMIT 10", "Validation Input")
dbutils.widgets.text("validation_thresholds_loader_function", "validation_thresholds", "Validation Thresholds Loader Function")
dbutils.widgets.text("evaluator_config_loader_function", "evaluator_config", "Evaluator Config Loader Function")
dbutils.widgets.text("model_name", "cvt_tumor_classifier", label="Full (Three-Level) Model Name")
dbutils.widgets.text("model_version", "5", "Candidate Model Version")
dbutils.widgets.text("env", "dev", "model environment")


# COMMAND ----------

print(
    "Currently model validation is not supported for models registered with feature store. Please refer to "
    "issue https://github.com/databricks/mlops-stacks/issues/70 for more details."
)

run_mode = dbutils.widgets.get("run_mode").lower()
assert run_mode == "disabled" or run_mode == "dry_run" or run_mode == "enabled"

if run_mode == "disabled":
    print(
        "Model validation is in DISABLED mode. Exit model validation without blocking model deployment."
    )
    dbutils.notebook.exit(0)
dry_run = run_mode == "dry_run"

if dry_run:
    print(
        "Model validation is in DRY_RUN mode. Validation threshold validation failures will not block model deployment."
    )
else:
    print(
        "Model validation is in ENABLED mode. Validation threshold validation failures will block model deployment."
    )

# COMMAND ----------

import importlib
import mlflow
import os
import tempfile
import traceback

from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks")

# set experiment
experiment_name = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(experiment_name)

env = dbutils.widgets.get("env")

# set model evaluation parameters that can be inferred from the job
model_uri = dbutils.jobs.taskValues.get("TrainTask", "model_uri", debugValue="")
model_name = dbutils.jobs.taskValues.get("TrainTask", "model_name", debugValue="")
model_version = dbutils.jobs.taskValues.get("TrainTask", "model_version", debugValue="1")
run_id = dbutils.jobs.taskValues.get("TrainTask", "run_id", debugValue="0")

if model_uri == "":
    model_name = dbutils.widgets.get("model_name")
    model_version = dbutils.widgets.get("model_version")
    model_uri = "models:/" + model_name + "/" + model_version

baseline_model_uri = "models:/" + model_name + "@Champion"

evaluators = "default"
assert model_uri != "", "model_uri notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
assert model_version != "", "model_version notebook parameter must be specified"

# COMMAND ----------

# take input
enable_baseline_comparison = dbutils.widgets.get("enable_baseline_comparison")
assert enable_baseline_comparison == "true" or enable_baseline_comparison == "false"
enable_baseline_comparison = enable_baseline_comparison == "true"

validation_input = dbutils.widgets.get("validation_input")
assert validation_input
data = spark.sql(validation_input)

model_type = dbutils.widgets.get("model_type")
targets = dbutils.widgets.get("targets")

assert model_type
assert targets

custom_metrics_loader_function_name = dbutils.widgets.get("custom_metrics_loader_function")
validation_thresholds_loader_function_name = dbutils.widgets.get("validation_thresholds_loader_function")
evaluator_config_loader_function_name = dbutils.widgets.get("evaluator_config_loader_function")
assert custom_metrics_loader_function_name
assert validation_thresholds_loader_function_name
assert evaluator_config_loader_function_name
custom_metrics_loader_function = getattr(
    importlib.import_module("validation"), custom_metrics_loader_function_name
)
validation_thresholds_loader_function = getattr(
    importlib.import_module("validation"), validation_thresholds_loader_function_name
)
evaluator_config_loader_function = getattr(
    importlib.import_module("validation"), evaluator_config_loader_function_name
)
custom_metrics = custom_metrics_loader_function()
validation_thresholds = validation_thresholds_loader_function()
evaluator_config = evaluator_config_loader_function()

# COMMAND ----------

# helper methods
def get_run_link(run_info):
    return "[Run](#mlflow/experiments/{0}/runs/{1})".format(
        run_info.experiment_id, run_info.run_id
    )


def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    return mlflow.get_run(run_id=version.run_id)


def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"


def generate_description(training_run):
    return (
        None
        if not training_run
        else "Model Training Details: {0}\n".format(get_run_link(training_run.info))
    )


def log_to_model_description(run, success):
    run_link = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description
    status = "SUCCESS" if success else "FAILURE"
    if description != "":
        description += "\n\n---\n\n"
    description += "Model Validation Status: {0}\nValidation Details: {1}".format(
        status, run_link
    )
    client.update_model_version(
        name=model_name, version=model_version, description=description
    )

# COMMAND ----------

import mlflow
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks')

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
seqAsVector = udf(lambda x : Vectors.dense(x), returnType=VectorUDT())

# COMMAND ----------

data = spark.sql("SELECT id,features,label FROM mlops_demo.cvt.patch_features LIMIT 10")
eval_data = data.withColumn("dense_features", seqAsVector("features")).cache()

# COMMAND ----------

eval_data.count()

# COMMAND ----------

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)

# COMMAND ----------

result_df = model.transform(eval_data).select("prediction","label")

# COMMAND ----------

display(result_df)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction") 

f1_score = evaluator.evaluate(result_df, {evaluator.metricName: 'f1'})
precision = evaluator.evaluate(result_df,{evaluator.metricName: 'weightedPrecision'})
recall = evaluator.evaluate(result_df, {evaluator.metricName: 'weightedRecall'})
accuracy = evaluator.evaluate(result_df, {evaluator.metricName: 'accuracy'})

print('F1-Score ', f1_score)
print('Precision ', precision)
print('Recall ', recall)
print('Accuracy ', accuracy)

eval_result_metrics = {
    "f1_score":f1_score,
    "precision":precision,
    "recall":recall,
    "accuracy":accuracy
}

# COMMAND ----------

from mlflow.models import MetricThreshold

def validate_thresholds(thresholds, eval_results):
    for t in thresholds:
        threshold = thresholds[t].threshold        
        result_value = eval_results[t]
        if thresholds[t].higher_is_better:
            if result_value < threshold:
                raise Exception(f"Evaluation failed for {t}")
        else:
            if result_value > threshold:
                raise Exception(f"Evaluation failed for {t}")

# COMMAND ----------

training_run = get_training_run(model_name, model_version)
run_name=generate_run_name(training_run)

# run evaluate
with mlflow.start_run(
    run_name=run_name,
    description=generate_description(training_run),
) as run, tempfile.TemporaryDirectory() as tmp_dir:
    
    try:
        validate_thresholds(validation_thresholds, eval_result_metrics)
        validation_thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
        
        with open(validation_thresholds_file, "w") as f:
            if validation_thresholds:
                for metric_name in validation_thresholds:
                    f.write(
                        "{0:30}  {1}\n".format(
                            metric_name, str(validation_thresholds[metric_name])
                        )
                    )
        mlflow.log_artifact(validation_thresholds_file)

        metrics_file = os.path.join(tmp_dir, "metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(
                "{0:30}  {1:30}  {2}\n".format("metric_name", "candidate", "baseline")
            )
            for metric in eval_result_metrics:
                candidate_metric_value = str(eval_result_metrics[metric])
                baseline_metric_value = "N/A"
                
                #if metric in eval_result.baseline_model_metrics:
                #    mlflow.log_metric(
                #        "baseline_" + metric, eval_result.baseline_model_metrics[metric]
                #    )
                #    baseline_metric_value = str(
                #        eval_result.baseline_model_metrics[metric]
                #    )
                f.write(
                    "{0:30}  {1:30}  {2}\n".format(
                        metric, candidate_metric_value, baseline_metric_value
                    )
                )
        mlflow.log_artifact(metrics_file)
        log_to_model_description(run, True)

        target_stage = "Production" if env == "production" else "Staging"
        
        print(f"Validation checks passed. Transitioning to {target_stage}")
        client.transition_model_version_stage(model_name,version=model_version, stage=target_stage, archive_existing_versions=True)
        
    except Exception as err:
        log_to_model_description(run, False)
        error_file = os.path.join(tmp_dir, "error.txt")
        with open(error_file, "w") as f:
            f.write("Validation failed : " + str(err) + "\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)

        raise err
        

# COMMAND ----------


