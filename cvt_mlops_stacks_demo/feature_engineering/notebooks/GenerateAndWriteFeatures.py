# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

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

# Feature table to store the computed features.
dbutils.widgets.text("patch_output_table_name","mlops_demo.cvt.patch_features",label="Output Feature Table Name")

# Feature transform module name.
dbutils.widgets.text("patch_features_module", "feature-extraction", label="Patch features transform notebooks")

# Primary Keys columns for the feature table;
dbutils.widgets.text("patch_primary_keys", "id",label="Primary keys columns for the patch feature table, comma separated.")

# COMMAND ----------

# DBTITLE 1,Define input and output variables
env = dbutils.widgets.get("env")
user_uuid = dbutils.widgets.get("user_uuid")

##notebook params for patch feature
patch_output_table_name = dbutils.widgets.get("patch_output_table_name")
patch_primary_keys = dbutils.widgets.get("patch_primary_keys")
patch_notebook = dbutils.widgets.get("patch_features_module")
assert patch_output_table_name != "", "output_table_name notebook parameter must be specified"

# Extract database name. Needs to be updated for Unity Catalog.
patch_output_catalog = patch_output_table_name.split(".")[0]
patch_output_database = patch_output_table_name.split(".")[1]


# COMMAND ----------

# MAGIC %md
# MAGIC ### Patch Generation

# COMMAND ----------

# DBTITLE 1,Create database.
spark.sql(f"CREATE DATABASE IF NOT EXISTS {patch_output_catalog}.{patch_output_database}")

# COMMAND ----------

# DBTITLE 1,Invoke patch generation notebook
dbutils.notebook.run(f"../features/{patch_notebook}",
                     600,
                     {
                         "env": env,
                         "user_uuid": user_uuid,
                         "patch_output_table_name": patch_output_table_name,
                         "patch_primary_keys": patch_primary_keys
                     })

# COMMAND ----------


