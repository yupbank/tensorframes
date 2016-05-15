package org.tensorframes

import org.scalatest.FunSuite

import org.tensorframes.dsl._
import org.tensorframes.dsl.Implicits._
import org.tensorframes.impl.DebugRowOps

import org.apache.spark.Logging
import org.apache.spark.sql.Row

// Some basic operations that stress shape transforms mostly.
class BasicOperationsSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging with GraphScoping {
  lazy val sql = sqlContext
  import Shape.Unknown

  val ops = new DebugRowOps

  testGraph("Identity") {
    val df = make1(Seq(1.0, 2.0), "in")
    val p1 = placeholder[Double](Unknown) named "in"
    val out = identity(p1) named "out"
    val df2 = df.mapBlocks(out).select("in", "out")
    assert(df2.collect() === Array(Row(1.0, 1.0), Row(2.0, 2.0)))
  }

  testGraph("Simple add") {
    val df = sql.createDataFrame(Seq(1.0->1.1, 2.0->2.2)).toDF("a", "b")
    val a = placeholder[Double](Unknown) named "a"
    val b = placeholder[Double](Unknown) named "b"
    val out = a + b named "out"
    val df2 = df.mapBlocks(out).select("a", "b","out")
    assert(df2.collect() === Array(Row(1.0, 1.1, 2.1), Row(2.0, 2.2, 4.2)))
  }

  testGraph("Identity - 1 dim") {
    val df = make1(Seq(Seq(1.0), Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder[Double](Unknown, 1) named "in"
    val out = identity(p1) named "out"
    val df2 = adf.mapBlocks(out).select("in", "out")
    assert(df2.collect() === Array(Row(Seq(1.0), Seq(1.0)), Row(Seq(2.0), Seq(2.0))))
  }

  testGraph("Simple add - 1 dim") {
    val a = placeholder[Double](Unknown, 1) named "a"
    val b = placeholder[Double](Unknown, 1) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0)->Seq(1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = adf.mapBlocks(out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.1), Seq(2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  testGraph("Reduce - sum double") {
    val df = make1(Seq(1.0, 2.0), "x")
    val x1 = placeholder[Double]() named "x_1"
    val x2 = placeholder[Double]() named "x_2"
    val x = x1 + x2 named "x"
    val r = df.reduceRows(x)
    assert(r === Row(3.0))
  }

  testGraph("Reduce - sum int") {
    val df = make1(Seq(1, 2, 3, 4), "x")
    val x1 = placeholder[Int]() named "x_1"
    val x2 = placeholder[Int]() named "x_2"
    val x = x1 + x2 named "x"
    val r = df.reduceRows(x)
    assert(r === Row(10))
  }

  testGraph("Simple add row - id") {
    val df = make1(Seq(1.0, 2.0), "in")
    val p1 = placeholder[Double]() named "in"
    val out = identity(p1) named "out"
    val df2 = df.mapRows(out).select("in", "out")
    assert(df2.collect() === Array(Row(1.0, 1.0), Row(2.0, 2.0)))
  }

  testGraph("Simple add - one row") {
    val df = sql.createDataFrame(Seq(
      1.0->1.1,
      2.0->2.2)).toDF("a", "b")
    val a = placeholder[Double]() named "a"
    val b = placeholder[Double]() named "b"
    val out = a + b named "out"
    val df2 = df.mapRows(out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(1.0, 1.1, 2.1),
      Row(2.0, 2.2, 4.2)))
  }

  testGraph("Identity - 1 known dim") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder[Double](1) named "in"
    val out = identity(p1) named "out"
    val df2 = adf.mapRows(out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0), Seq(2.0))))
  }

  testGraph("Identity - 1 unknown dim") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder[Double](Unknown) named "in"
    val out = identity(p1) named "out"
    val df2 = adf.mapRows(out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0), Seq(2.0))))
  }

  testGraph("Identity - 1 dim and variable sizes") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0, 2.1)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder[Double](Unknown) named "in"
    val out = identity(p1) named "out"
    val df2 = adf.mapRows(out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0, 2.1), Seq(2.0, 2.1))))
  }

  testGraph("Simple add row - 1 dim") {
    val a = placeholder[Double](1) named "a"
    val b = placeholder[Double](1) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0)->Seq(1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = adf.mapRows(out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.1), Seq(2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  testGraph("Simple add row - 1 dim unknown rows") {
    val a = placeholder[Double](Unknown) named "a"
    val b = placeholder[Double](Unknown) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0, 1.0)->Seq(1.1, 1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    logInfo(s"df: \n${ops.explain(df)}")
    val adf = ops.analyze(df)
    logInfo(s"adf: \n${ops.explain(adf)}")
    val df2 = adf.mapRows(out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0, 1.0), Seq(1.1, 1.1), Seq(2.1, 2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  testGraph("Reduce block - sum double") {
    val df = make1(Seq(1.0, 2.0), "x")
    val x1 = placeholder[Double](Unknown) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = df.reduceBlocks(x)
    assert(r === Row(3.0))
  }

  testGraph("Reduce block - sum double with extra column") {
    val df = sql.createDataFrame(Seq(
      ("1", 1.0),
      ("2", 1.1),
      ("3", 2.0))).toDF("key2", "x")
    val x1 = placeholder[Double](Unknown) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = df.reduceBlocks(x)
    assert(r === Row(4.1))
  }

  testGraph("Reduce block - sum double with extra column and fixed block size") {
    // coalesce fails on old spark version, you need to explicitly build the RDD
    val rdd = sqlContext.sparkContext.makeRDD(Seq(Tuple1(1.0), Tuple1(2.0)), 2)
    val df = sqlContext.createDataFrame(rdd).toDF("x")
      .analyze()
    val x1 = placeholder[Double](Unknown) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = df.reduceBlocks(x)
    assert(r === Row(3.0))
  }
  
  testGraph("Aggregate over rows") {
    val df = sql.createDataFrame(Seq(
      (1, 1.0),
      (1, 1.1),
      (2, 2.0))).toDF("key", "x")
    val x1 = placeholder[Double](Unknown) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val df2 = df.groupBy("key").aggregate(x).select("key", "x")
    df2.printSchema()
    assert(df2.collect() === Array(Row(1, 2.1), Row(2, 2.0)))
  }
}
