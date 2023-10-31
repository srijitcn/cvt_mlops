# Databricks notebook source
##################################################################################
# Generate and Write Features Notebook
#
# This notebook can be used to generate and write features to a Databricks Feature Store table.
# It is configured and can be executed as the tasks in the write_feature_table_job workflow defined under
# ``cvt_mlops_stacks_demo/resources/feature-engineering-workflow-resource.yml``
#
# Parameters:
#
# * env (required)                - The deploy environment
# * user_uuid (required)          - The deploy environment
# * output_table_name (required)  - Fully qualified schema + Delta table name for the feature table where the features
# *                                 will be written to. Note that this will create the Feature table if it does not
# *                                 exist.
# * primary_keys (required)       - A comma separated string of primary key columns of the output feature table.
# *
# * timestamp_column (optional)   - Timestamp column of the input data. Used to limit processing based on
# *                                 date ranges. This column is used as the timestamp_key column in the feature table.
# * input_start_date (optional)   - Used to limit feature computations based on timestamp_column values.
# * input_end_date (optional)     - Used to limit feature computations based on timestamp_column values.
# *
# * features_transform_module (required) - Python module containing the feature transform logic.
##################################################################################


# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# env
dbutils.widgets.text("env","dev",label="Target Env")

# user_uuid
dbutils.widgets.text("user_uuid","8243",label="User UUID")

# Input start date.
dbutils.widgets.text("input_start_date", "", label="Input Start Date")

# Input end date.
dbutils.widgets.text("input_end_date", "", label="Input End Date")

# Timestamp column. Will be used to filter input start/end dates.
# This column is also used as a timestamp key of the feature table.
dbutils.widgets.text(
    "timestamp_column", "tpep_pickup_datetime", label="Timestamp column"
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "output_table_name",
    "mlops_demo.cvt.patch_features",
    label="Output Feature Table Name",
)

# Feature transform module name.
dbutils.widgets.text(
    "features_transform_module", "patch_feature_extraction", label="Features transform file."
)
# Primary Keys columns for the feature table;
dbutils.widgets.text(
    "primary_keys",
    "id",
    label="Primary keys columns for the feature table, comma separated.",
)

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../features


# COMMAND ----------

# DBTITLE 1,Define input and output variables
env = dbutils.widgets.get("env")
user_uuid = dbutils.widgets.get("user_uuid")

output_table_name = dbutils.widgets.get("output_table_name")
input_start_date = dbutils.widgets.get("input_start_date")
input_end_date = dbutils.widgets.get("input_end_date")
ts_column = dbutils.widgets.get("timestamp_column")
features_module = dbutils.widgets.get("features_transform_module")
pk_columns = dbutils.widgets.get("primary_keys")

assert output_table_name != "", "output_table_name notebook parameter must be specified"

# Extract database name. Needs to be updated for Unity Catalog.
output_catalog = output_table_name.split(".")[0]
output_database = output_table_name.split(".")[1]


# COMMAND ----------

# DBTITLE 1,Create database.
spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_catalog}.{output_database}")

# COMMAND ----------

# DBTITLE 1,Compute features.
# Compute the features. This is done by dynamically loading the features module.
from importlib import import_module

mod = import_module(features_module)
compute_features_fn = getattr(mod, "compute_features_fn")

features_df = compute_features_fn(
    env = env,
    user_uuid = user_uuid
)

# COMMAND ----------

# DBTITLE 1, Write computed features.
from databricks import feature_store

fs = feature_store.FeatureStoreClient()


# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.
fs.create_table(
    name=output_table_name,
    primary_keys=[x.strip() for x in pk_columns.split(",")],
    timestamp_keys=[ts_column],
    df=features_df,
)

# Write the computed features dataframe.
fs.write_table(
    name=output_table_name,
    df=features_df,
    mode="merge",
)

dbutils.notebook.exit(0)

# COMMAND ----------


