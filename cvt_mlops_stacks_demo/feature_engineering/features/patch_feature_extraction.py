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

# MAGIC %run ./definitions

# COMMAND ----------

import json
import os
from pprint import pprint
from pyspark.sql.functions import *

def compute_features_fn(env, user_uuid):
    
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
    
    project_name='digital-pathology'
    config_path=f"/dbfs/FileStore/mlops_digital_pathology/{env}/{user_uuid}_{project_name}_configs.json"

    try:
        with open(config_path,'rb') as f:
            settings = json.load(f)
    except FileNotFoundError:
        print('please run ./config notebook and try again')
    assert False

    #set paths
    WSI_PATH=settings['data_path']
    BASE_PATH=settings['base_path']
    IMG_PATH = settings['img_path']
    ANNOTATION_PATH = BASE_PATH+"/annotations"

    #define params
    PATCH_SIZE=settings['patch_size']
    LEVEL=settings['level']
    
    annotation_df=spark.read.load(f'{ANNOTATION_PATH}/delta/patch_labels').withColumn('imid',concat_ws('-',col('sid'),col('x_center'),col('y_center')))

    #Create a dataframe of processed patches

    patch_df= (
        spark.read.format('binaryFile').load(f'{IMG_PATH}/*/*/*.jpg')
            .repartition(32)
            .withColumn('imid',regexp_extract('path','(\\w+)_(\\d+)-(\\d+)-(\\d+)', 0))
    )

    #Create a dataframe of processed patches with metadata
    dataset_df = (
        annotation_df
            .join(patch_df,on='imid')
            .selectExpr('uuid() as id','sid as slide_id','x_center','y_center','label','content')
        )
    
    #Extract features from images
    features_df=dataset_df.select('*',featurize_raw_img_series_udf('content').alias('features'))

    return features_df

# COMMAND ----------


