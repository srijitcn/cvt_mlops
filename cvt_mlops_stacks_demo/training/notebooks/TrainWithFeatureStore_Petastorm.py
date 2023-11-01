# Databricks notebook source
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
            "content"
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
input_data = spark.table(feature_table).select("id","label").limit(5)

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
    input_data,
    feature_lookups=feature_lookups,
    label="label",
    exclude_columns=["features","id"],
)

# Load the TrainingSet into a dataframe
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, 
training_df.display()

# COMMAND ----------

df_train, df_val = training_df.randomSplit([0.9, 0.1], seed=12345)


# COMMAND ----------

spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///dbfs/tmp/petastorm/cache")

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

def transform_row(is_train, pd_batch):
  """
  The input and output of this function must be pandas dataframes.
  Do data augmentation for the training dataset only.
  """
  transformers = [transforms.Lambda(lambda x: Image.open(io.BytesIO(x)))]
  if is_train:
    transformers.extend([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
    ])
  else:
    transformers.extend([
      transforms.Resize(256),
      transforms.CenterCrop(224),
    ])
  transformers.extend([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])
  
  trans = transforms.Compose(transformers)
  
  pd_batch['features'] = pd_batch['content'].map(lambda x: trans(x).numpy())
  pd_batch = pd_batch.drop(labels=['content'], axis=1)
  return pd_batch

def get_transform_spec(is_train=True):
  # Note that the output shape of the `TransformSpec` is not automatically known by petastorm, 
  # so we need to specify the shape for new columns in `edit_fields` and specify the order of 
  # the output columns in `selected_fields`.
  return TransformSpec(partial(transform_row, is_train), 
                       edit_fields=[('features', np.float32, (3, 224, 224), False), ('label', np.float32, (1,0,0), False)], 
                       selected_fields=['features', 'label'])

# COMMAND ----------

# DBTITLE 1, Train model
def get_model(lr=0.001):
  # Load a MobileNetV2 model from torchvision
  model = torchvision.models.mobilenet_v2(pretrained=True)
  # Freeze parameters in the feature extraction layers
  for param in model.parameters():
    param.requires_grad = False
    
  # Add a new classifier layer for transfer learning
  num_ftrs = model.classifier[1].in_features
  # Parameters of newly constructed modules have requires_grad=True by default
  model.classifier[1] = torch.nn.Linear(num_ftrs, 1)
  
  return model


# COMMAND ----------

BATCH_SIZE = 1
NUM_EPOCHS = 5

# COMMAND ----------

def train(model, 
          criterion, 
          optimizer, 
          scheduler,
          train_dataloader_iter, 
          steps_per_epoch, 
          epoch,
          device):
    
  model.train()  # Set model to training mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over the data for one epoch.
  for step in range(steps_per_epoch):
    pd_batch = next(train_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label'].to(device)
    
    # Track history in training
    with torch.set_grad_enabled(True):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward
      outputs = model(inputs)

      print("###############")
      print(inputs)
      print(outputs)
      print(labels)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

      # backward + optimize
      loss.backward()
      optimizer.step()

    # statistics
    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)
  
  scheduler.step()

  epoch_loss = running_loss / (steps_per_epoch * BATCH_SIZE)
  epoch_acc = running_corrects.double() / (steps_per_epoch * BATCH_SIZE)

  print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

def evaluate(model, criterion, 
             val_dataloader_iter, 
             validation_steps,
             device, 
             metric_agg_fn=None):
  model.eval()  # Set model to evaluate mode

  # statistics
  running_loss = 0.0
  running_corrects = 0

  # Iterate over all the validation data.
  for step in range(validation_steps):
    pd_batch = next(val_dataloader_iter)
    inputs, labels = pd_batch['features'].to(device), pd_batch['label'].to(device)

    # Do not track history in evaluation to save memory
    with torch.set_grad_enabled(False):
      # forward
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)

    # statistics
    running_loss += loss.item()
    running_corrects += torch.sum(preds == labels.data)
  
  # The losses are averaged across observations for each minibatch.
  epoch_loss = running_loss / validation_steps
  epoch_acc = running_corrects.double() / (validation_steps * BATCH_SIZE)
  
  # metric_agg_fn is used in the distributed training to aggregate the metrics on all workers
  if metric_agg_fn is not None:
    epoch_loss = metric_agg_fn(epoch_loss, 'avg_loss')
    epoch_acc = metric_agg_fn(epoch_acc, 'avg_acc')

  print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
  return epoch_loss, epoch_acc

# COMMAND ----------

def train_and_evaluate(lr=0.001):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  model = get_model(lr=lr)
  model = model.to(device)

  criterion = torch.nn.CrossEntropyLoss()

  # Only parameters of final layer are being optimized.
  optimizer = torch.optim.SGD(model.classifier[1].parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  
  with converter_train.make_torch_dataloader(transform_spec=get_transform_spec(is_train=True), 
                                             batch_size=BATCH_SIZE) as train_dataloader, \
       converter_val.make_torch_dataloader(transform_spec=get_transform_spec(is_train=False), 
                                           batch_size=BATCH_SIZE) as val_dataloader:
    
    train_dataloader_iter = iter(train_dataloader)
    steps_per_epoch = len(converter_train) // BATCH_SIZE
    
    val_dataloader_iter = iter(val_dataloader)
    validation_steps = max(1, len(converter_val) // BATCH_SIZE)
    
    for epoch in range(NUM_EPOCHS):
      print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
      print('-' * 10)

      train_loss, train_acc = train(model, criterion, optimizer, exp_lr_scheduler, 
                                              train_dataloader_iter, steps_per_epoch, epoch, 
                                              device)
      val_loss, val_acc = evaluate(model, criterion, val_dataloader_iter, validation_steps, device)

  return val_loss
  

# COMMAND ----------

loss = train_and_evaluate()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Log the trained model with MLflow and package it with feature lookup information.
fs.log_model(
    model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
