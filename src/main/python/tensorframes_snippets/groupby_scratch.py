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


data = [Row(x=float(x)) for x in range(5)]
df = sqlContext.createDataFrame(data)
with tf.Graph().as_default() as g:
    # The placeholder that corresponds to column 'x'
    x = tf.placeholder(tf.double, shape=[None], name="x")
    # The output that adds 3 to x
    z = tf.add(x, 3, name='z')
    # The resulting dataframe
    df2 = tfs.map_blocks(z, df)

df2.show()

