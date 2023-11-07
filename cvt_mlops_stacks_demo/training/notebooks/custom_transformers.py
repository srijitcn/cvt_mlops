from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol,HasOutputCol, Param, Params
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf

class DenseVectorizer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, outputCol=None):
        super(DenseVectorizer, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)  
        self.setParams(inputCol=inputCol, outputCol=outputCol)  

    def setParams(self, inputCol=None, outputCol=None):
        return self._set(inputCol=inputCol, outputCol=outputCol)  

    def _transform(self, df) -> DataFrame:
        seqAsVector = udf(lambda x : Vectors.dense(x), returnType=VectorUDT())
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        return df.withColumn(output_col, seqAsVector(input_col))