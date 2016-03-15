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
# TODO(tjh) make a test out of this, it found some bugs
# - non numeric columns (string)
# - unused columns
# - output that has a child
col_name = "x"
col_key = "key"
with tf.Graph().as_default() as g:
    x = tfs.block(df, col_name)
    invs = tf.inv(tf.to_double(x), name="invs")
    df2 = tfs.map_blocks([invs, tf.ones_like(invs, name="count")], df)


# The geometric mean
gb = df2.select(col_key, "invs", "count").groupBy("key")
with tf.Graph().as_default() as g:
    x_input = tfs.block(df2, "invs", tf_name="invs_input")
    count_input = tfs.block(df2, "invs", tf_name="count_input")
    x = tf.reduce_sum(x_input, [0], name='invs')
    count = tf.reduce_sum(count_input, [0], name='count')
    df3 = tfs.aggregate([x, count], gb)

with tf.Graph().as_default() as g:
    invs = tfs.block(df2, "invs")
    count = tfs.block(df2, "count")
    geom_mean = tf.div(tf.to_double(count), invs,  name = "harmonic_mean")
    df4 = tfs.map_blocks(geom_mean, df3).select("key", "harmonic_mean")

df4.collect()
