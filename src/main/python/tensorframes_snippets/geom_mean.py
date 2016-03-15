import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
import tensorframes as tfs
import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType

from tensorframes.core import _java_api
japi = _java_api()
_java_api().initialize_logging()


# The input data
data = [Row(x=float(x), key=str(x / 3)) for x in range(1, 6)]
df = sqlContext.createDataFrame(data)

# The geometric mean:
col_name = "x"
col_key = "key"
with tf.Graph().as_default() as g:
    x = tfs.block(df, col_name)
    invs = tf.inv(tf.to_double(x), name="invs")
    df2 = tfs.map_blocks([invs, tf.ones_like(invs, name="count")], df)


# The geometric mean
gb = df.withColumn("count", lit(1)).groupBy("key")
with tf.Graph().as_default() as g:
    x_input = tf.placeholder(tf.double, shape=[None], name="x_input")
    count_input = tf.placeholder(tf.int32, shape=[None], name="count_input")
    invs = tf.inv(x_input)
    x = tf.reduce_sum(invs, [0], name='x')
    count = tf.reduce_sum(count_input, [0], name='count')
    df2 = tfs.aggregate([x, count], gb)

with tf.Graph().as_default() as g:
    x = tf.placeholder(tf.double, shape=[None], name="x")
    count = tf.placeholder(tf.int32, shape=[None], name="count")
    geom_mean = tf.div(tf.inv(x), tf.to_double(count), name = "geom_mean")
    df3 = tfs.map_blocks(geom_mean, df2).select("key", "geom_mean")

