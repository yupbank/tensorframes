import tensorframes as tfs
import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType

from tensorframes.core import _java_api
japi = _java_api()
_java_api().initialize_logging()

data = [Row(x=float(x), key=str(x / 3)) for x in range(1, 6)]
df = sqlContext.createDataFrame(data)
tfs.block(df, "x")

data = [Row(x=float(x), key=str(x / 3)) for x in range(1, 6)]
df = sqlContext.createDataFrame(data)
gb = df.groupBy("key")
with tf.Graph().as_default() as g:
    x_input = tfs.block(df, "x", tf_name="x_input")
    x = tf.reduce_sum(x_input, [0], name='x')
    df2 = tfs.aggregate(x, gb)
