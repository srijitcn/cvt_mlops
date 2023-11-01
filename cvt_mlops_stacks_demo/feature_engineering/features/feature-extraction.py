# Databricks notebook source
# MAGIC %md
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/digital-pathology. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/digital-pathology.

# COMMAND ----------

# MAGIC %md
# MAGIC # Distributed feature extraction
# MAGIC In this notebook we use spark's `pandas_udfs` to efficiently distribute feature extraction process. The extracted features are then can be used to visually inspect the structure of extracted patches.
# MAGIC
# MAGIC
# MAGIC <img src="https://cloud.google.com/tpu/docs/images/inceptionv3onc--oview.png">
# MAGIC
# MAGIC We use embeddings based on a pre-trained deep neural network (in this example, [InceptionV3](https://arxiv.org/abs/1512.00567)) to extract features from each patch.
# MAGIC Associated methods for feature extraction are defined within `./definitions` notebook in this package.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Initial Configuration

# COMMAND ----------

dbutils.widgets.text("env","dev",label="Target Env")

# user_uuid
dbutils.widgets.text("user_uuid","8243",label="User UUID")

# Feature table to store the computed features.
dbutils.widgets.text("patch_output_table_name","mlops_demo.cvt.patch_features",label="Output Feature Table Name")

# Primary Keys columns for the feature table;
dbutils.widgets.text("patch_primary_keys", "id",label="Primary keys columns for the patch feature table, comma separated.")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al /dbfs/FileStore/

# COMMAND ----------

env = dbutils.widgets.get("env")
user_uid = dbutils.widgets.get("user_uuid")
patch_output_table_name = dbutils.widgets.get("patch_output_table_name")
patch_primary_keys = dbutils.widgets.get("patch_primary_keys")

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

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

from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,set paths
WSI_PATH=settings['data_path']
BASE_PATH=settings['base_path']
IMG_PATH = settings['img_path']
ANNOTATION_PATH = BASE_PATH+"/annotations"

# COMMAND ----------

# DBTITLE 1,define parameters
PATCH_SIZE=settings['patch_size']
LEVEL=settings['level']

# COMMAND ----------

annotation_df=spark.read.load(f'{ANNOTATION_PATH}/delta/patch_labels').withColumn('imid',concat_ws('-',col('sid'),col('x_center'),col('y_center')))
display(annotation_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create a dataframe of processed patches
# MAGIC Now we create dataframe of processed patches with their associated annotations

# COMMAND ----------

# DBTITLE 1,Create a dataframe of images
patch_df= (
  spark.read.format('binaryFile').load(f'{IMG_PATH}/*/*/*.jpg').limit(1) #repartition(32)
  .withColumn('imid',regexp_extract('path','(\\w+)_(\\d+)-(\\d+)-(\\d+)', 0))
)

# COMMAND ----------

# DBTITLE 1,Create a dataframe of processed patches with metadata
dataset_df = (
  annotation_df
  .join(patch_df,on='imid')
  .selectExpr('uuid() as id','sid as slide_id','x_center','y_center','label','content')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extract features from images
# MAGIC Now that we have a dataframe of all patches, we pass each patch through the pre-trained model and use the networks output (embeddings) as features to be used for dimensionality reduction.

# COMMAND ----------

# MAGIC %run ./definitions

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we simply apply the `featurize_raw_img_series_udf` function which is defined in `./definitions` notebook to extracted embeddings from each image in a distributed fashion, using `pandas_udf` functionality within Apache Spark.

# COMMAND ----------

features_df=(dataset_df
             .select('*',featurize_raw_img_series_udf('content').alias('features'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see bellow, the resulting dataframe contains the content of each patch, as well as associated annotation and extracted features, all in one table.

# COMMAND ----------

# MAGIC %md
# MAGIC ##3. Create Feature Store in delta
# MAGIC Now to persist our results, we write the resulting dataframe into deltalake for future access.

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

features_df.cache().count()

# Create the feature table if it does not exist first.
# Note that this is a no-op if a table with the same name and schema already exists.
fs.create_table(
    name=patch_output_table_name,
    primary_keys=[x.strip() for x in patch_primary_keys.split(",")],
    #timestamp_keys=[ts_column],
    df=features_df,
)

# COMMAND ----------

# Write the computed features dataframe.
fs.write_table(
    name=patch_output_table_name,
    df=features_df,
    mode="merge",
)


# COMMAND ----------


