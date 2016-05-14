package org.tensorframes

import org.scalatest.FunSuite
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.test.dsl._

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType}

// Some basic operations that stress shape transforms mostly.
class BasicOperationsSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging {
  lazy val sql = sqlContext
  import Shape.Unknown

  val ops = new DebugRowOps

  test("Identity") {
    val df = make1(Seq(1.0, 2.0), "in")
    val p1 = placeholder(DoubleType, Shape(Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapBlocks(df, out).select("in", "out")
    assert(df2.collect() === Array(Row(1.0, 1.0), Row(2.0, 2.0)))
  }

  test("Simple add") {
    val df = sql.createDataFrame(Seq(1.0->1.1, 2.0->2.2)).toDF("a", "b")
    val a = placeholder(DoubleType, Shape(Unknown)) named "a"
    val b = placeholder(DoubleType, Shape(Unknown)) named "b"
    val out = a + b named "out"
    val df2 = ops.mapBlocks(df, out).select("a", "b","out")
    assert(df2.collect() === Array(Row(1.0, 1.1, 2.1), Row(2.0, 2.2, 4.2)))
  }

  test("Identity - 1 dim") {
    val df = make1(Seq(Seq(1.0), Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(DoubleType, Shape(Unknown, 1)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapBlocks(adf, out).select("in", "out")
    assert(df2.collect() === Array(Row(Seq(1.0), Seq(1.0)), Row(Seq(2.0), Seq(2.0))))
  }

  test("Simple add - 1 dim") {
    val a = placeholder(DoubleType, Shape(Unknown, 1)) named "a"
    val b = placeholder(DoubleType, Shape(Unknown, 1)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0)->Seq(1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = ops.mapBlocks(adf, out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.1), Seq(2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  test("Reduce - sum double") {
    val df = make1(Seq(1.0, 2.0), "x")
    val x1 = placeholder(DoubleType, Shape.empty) named "x_1"
    val x2 = placeholder(DoubleType, Shape.empty) named "x_2"
    val x = x1 + x2 named "x"
    val r = ops.reduceRows(df, x)
    assert(r === Row(3.0))
  }

  test("Reduce - sum int") {
    val df = make1(Seq(1, 2, 3, 4), "x")
    val x1 = placeholder(IntegerType, Shape.empty) named "x_1"
    val x2 = placeholder(IntegerType, Shape.empty) named "x_2"
    val x = x1 + x2 named "x"
    val r = ops.reduceRows(df, x)
    assert(r === Row(10))
  }

  test("Simple add row - id") {
    val df = make1(Seq(1.0, 2.0), "in")
    val p1 = placeholder(DoubleType, Shape.empty) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(df, out).select("in", "out")
    assert(df2.collect() === Array(Row(1.0, 1.0), Row(2.0, 2.0)))
  }

  test("Simple add - one row") {
    val df = sql.createDataFrame(Seq(
      1.0->1.1,
      2.0->2.2)).toDF("a", "b")
    val a = placeholder(DoubleType, Shape.empty) named "a"
    val b = placeholder(DoubleType, Shape.empty) named "b"
    val out = a + b named "out"
    val df2 = ops.mapRows(df, out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(1.0, 1.1, 2.1),
      Row(2.0, 2.2, 4.2)))
  }

  test("Identity - 1 known dim") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(DoubleType, Shape(1)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0), Seq(2.0))))
  }

  test("Identity - 1 unknown dim") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(DoubleType, Shape(Shape.Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0), Seq(2.0))))
  }

  test("Identity - 1 dim and variable sizes") {
    val df = make1(Seq(
      Seq(1.0),
      Seq(2.0, 2.1)), "in")
    val adf = ops.analyze(df)
    val p1 = placeholder(DoubleType, Shape(Shape.Unknown)) named "in"
    val out = op_id(p1) named "out"
    val df2 = ops.mapRows(adf, out).select("in", "out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.0)),
      Row(Seq(2.0, 2.1), Seq(2.0, 2.1))))
  }

  test("Simple add row - 1 dim") {
    val a = placeholder(DoubleType, Shape(1)) named "a"
    val b = placeholder(DoubleType, Shape(1)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0)->Seq(1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    val adf = ops.analyze(df)
    val df2 = ops.mapRows(adf, out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0), Seq(1.1), Seq(2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  test("Simple add row - 1 dim unknown rows") {
    val a = placeholder(DoubleType, Shape(Unknown)) named "a"
    val b = placeholder(DoubleType, Shape(Unknown)) named "b"
    val out = a + b named "out"

    val df = sql.createDataFrame(Seq(
      Seq(1.0, 1.0)->Seq(1.1, 1.1),
      Seq(2.0)->Seq(2.2))).toDF("a", "b")
    logInfo(s"df: \n${ops.explain(df)}")
    val adf = ops.analyze(df)
    logInfo(s"adf: \n${ops.explain(adf)}")
    val df2 = ops.mapRows(adf, out).select("a", "b","out")
    assert(df2.collect() === Array(
      Row(Seq(1.0, 1.0), Seq(1.1, 1.1), Seq(2.1, 2.1)),
      Row(Seq(2.0), Seq(2.2), Seq(4.2))))
  }

  test("Reduce block - sum double") {
    val df = make1(Seq(1.0, 2.0), "x")
    val x1 = placeholder(DoubleType, Shape(Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = ops.reduceBlocks(df, x)
    assert(r === Row(3.0))
  }

  test("Reduce block - sum double with extra column") {
    val df = sql.createDataFrame(Seq(
      ("1", 1.0),
      ("2", 1.1),
      ("3", 2.0))).toDF("key2", "x")
    val x1 = placeholder(DoubleType, Shape(Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val r = ops.reduceBlocks(df, x)
    assert(r === Row(4.1))
  }

  test("Aggregate over rows") {
    val df = sql.createDataFrame(Seq(
      (1, 1.0),
      (1, 1.1),
      (2, 2.0))).toDF("key", "x")
    val x1 = placeholder(DoubleType, Shape(Shape.Unknown)) named "x_input"
    val x = reduce_sum(x1, Seq(0)) named "x"
    val df2 = ops.aggregate(df.groupBy("key"), x).select("key", "x")
    df2.printSchema()
    assert(df2.collect() === Array(Row(1, 2.1), Row(2, 2.0)))
  }
}
