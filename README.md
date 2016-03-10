# tensorframes

Experimental TF binding for Scala and Spark.

TensorFrames (TensorFlow on Spark Dataframes) lets you manipulate Spark's DataFrames with TensorFlow programs.

> This package is highly experimental and is provided as a technical preview only.


> This package only supports linux 64bit platforms as a target. Contributions are welcome for other platforms.

Officially supported Spark versions: 1.6+

## Requirements

 - A working version of Spark (1.6 or greater), available locally through the `SPARK_HOME` variable.

 - java version >= 7
 
 - python >= 2.7 . Python 3+ should work but has not been tested.
 
 - (Optional) the python TensorFlow package if you want to use the python interface. See the 
 [official instructions](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#download-and-setup)
  on how to get the latest release of TensorFlow.

Additionally, if you want to run unit tests for python, you need the following dependencies:

 - nose >= 1.3 


## How to compile and install

There is no official release yet and you have to compile the code in order to use it.
 The C++ bindings are already compiled though, so you should only have to deal with compiling
 the scala code. The recommended procedure is to use the assembly:

```bash
build/sbt assembly
```

## How to run in python

You must compile the assembly first.

Then, assuming that `SPARK_HOME` is set and that you are at the project root, 
you can use PySpark:

```bash
PYTHONPATH=$PWD/target/scala-2.11/tensorframes-assembly-0.1.0-SNAPSHOT.jar IPYTHON=1 $SPARK_HOME/bin/pyspark --jars $PWD/target/scala-2.11/tensorframes-assembly-0.1.0-SNAPSHOT.jar
```

Here is a small program that uses Tensorflow to add 3 to an existing column.

```python
import tensorflow as tf
import tensorframes as tfs
from pyspark.sql import Row

data = [Row(x=float(x)) for x in range(10)]
df = sqlContext.createDataFrame(data)
with tf.Graph().as_default() as g:
    # The placeholder that corresponds to column 'x'
    x = tf.placeholder(tf.double, shape=[None], name="x")
    # The output that adds 3 to x
    z = tf.add(x, 3, name='z')
    # The resulting dataframe
    df2 = tfs.map_blocks(z, df)

# The transform is lazy as for most DataFrame operations. This will trigger it:
df2.collect()

# Notice that z is an extra column next to x

# [Row(z=3.0, x=0.0),
#  Row(z=4.0, x=1.0),
#  Row(z=5.0, x=2.0),
#  Row(z=6.0, x=3.0),
#  Row(z=7.0, x=4.0),
#  Row(z=8.0, x=5.0),
#  Row(z=9.0, x=6.0),
#  Row(z=10.0, x=7.0),
#  Row(z=11.0, x=8.0),
#  Row(z=12.0, x=9.0)]
```

The second example shows the block-wise reducing operations: we compute the sum of a field containing 
vectors of integers, working with blocks of rows for more efficient processing.

```python
# Build a DataFrame of vectors
data = [Row(y=[float(y), float(-y)]) for y in range(10)]
df = sqlContext.createDataFrame(data)
# Because the dataframe contains vectors, we need to analyze it first to find the
# dimensions of the vectors.
df2 = tfs.analyze(df)

# The information gathered by TF can be printed to check the content:
tfs.print_schema(df2)
# TF has inferred that y contains vectors of size 2
# root
#  |-- y: array (nullable = false) DoubleType[?,2]

# Let's use the analyzed dataframe to compute the sum and the elementwise minimum 
# of all the vectors:
# First, let's make a copy of the 'y' column. This will be very cheap in Spark 2.0+
df3 = df2.select(df2.y, df2.y.alias("z"))
with tf.Graph().as_default() as g:
    # The placeholder that corresponds to column 'y'. Note the special name:
    y_input = tf.placeholder(tf.double, shape=[None, 2], name="y_input")
    z_input = tf.placeholder(tf.double, shape=[None, 2], name="z_input")
    y = tf.reduce_sum(y_input, [0], name='y')
    z = tf.reduce_min(z_input, [0], name='z')
    # The resulting dataframe
    (data_sum, data_min) = tfs.reduce_blocks([y, z], df3)

print data_sum
# [45.0, -45.0]
print data_min
# [0.0, -9.0]
```

*Notes*

Note the scoping of the graphs above. This is important because TensorFrames finds which 
DataFrame column to feed to TensorFrames based on the placeholders of the graph. Also, it is 
 good practice to keep small graphs when sending them to Spark.
 
For small tensors (scalars and vectors), TensorFrames usually infers the shapes of the 
tensors without requiring a preliminary analysis. If it cannot do it, an error message will 
indicate it.

Look at the python documentation of the TensorFrames package to see what methods are available.


## How to run in Scala

The scala support is a bit more limited than python. In scala, operations can be loaded from 
 an existing graph defined in the ProtocolBuffers format, or using a simple scala DSL. The
 Scala DSL only features a very limited subset of TensorFlow transforms. It is very easy to extend
 though, so other transforms could be added without much effort in the future.

Assuming that SPARK_HOME is set and that you are in the root directory of the project:

```bash
$SPARK_HOME/bin/spark-shell --jars $PWD/target/scala-2.11/tensorframes-assembly-0.1.0-SNAPSHOT.jar
```

Here is a simple program to add two columns together:

```scala
import org.tensorframes.test.dsl._
import org.tensorframes.impl.{DebugRowOps => ops}
import org.tensorframes._
import org.apache.spark.sql.types._

val df = sqlContext.createDataFrame(Seq(1.0->1.1, 2.0->2.2)).toDF("a", "b")

val a = placeholder(DoubleType, Shape.empty) named "a"
val b = placeholder(DoubleType, Shape.empty) named "b"
val out = a + b named "out"
val df2 = ops.mapRows(df, out).select("a", "b","out")

// The transform is all lazy at this point, let's execute it with collect:
df2.collect()
// res0: Array[org.apache.spark.sql.Row] = Array([1.0,1.1,2.1], [2.0,2.2,4.2])   
```

## How to change the version of TensorFlow

By default, TensorFrames features a relatively stable version of TensorFlow that is optimized 
for build sizes and for CPUs. If you want to change the internal version being used, you should
check the [tensorframes-artifacts](https://github.com/tjhunter/tensorframes-artifacts) project. 
That project contains scripts to build the proper jar files. 
