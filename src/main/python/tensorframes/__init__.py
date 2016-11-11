"""
TensorFrames, a TensorFlow binding library for Scala/Spark.

This modules provides methods to manipulate Spark DataFrames with TensorFlow numerical graphs.

Core API
--------

The API lets users manipulate Spark dataframes using standard big data operations such as map,
reduce and aggregate. These operations come usually in two flavors: block and row. In a block
operation, multiple rows of data are manipulated at once (and the TensorFlow program that
expresses this transformation must accept these blocks). In a row operation, each row is processed
one at a time.

The most important operations available in this module are:

 - map_rows: adds extra columns one row at time
 - map_blocks: adds extra columns block by block
 - reduce_rows: applies a transform on pairs of rows until one row is left
 - reduce_blocks: applies a transform on blocks or rows until one row is left
 - aggregate: performs aggregation of blocks of rows based on a key index

TensorFrames needs sometimes some extra information about the shape of the numerical tensors in the
dataframe. These two methods extract and query this information:

 - analyze: perform an shape analysis of all the numerical data in a dataframe
 - print_schema: prints the schema of a dataframe, with extra numerical information

Each method contains extensive documentation about their usage. More documentation can be found
on the project wiki.

"""
from __future__ import absolute_import

__version__ = "2.0.0"

from .core import *
