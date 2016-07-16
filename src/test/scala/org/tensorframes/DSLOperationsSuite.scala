package org.tensorframes

import org.scalatest.FunSuite
import org.tensorframes.dsl._
import org.tensorframes.dsl.Implicits._

import org.tensorframes.Logging
import org.apache.spark.sql.Row

class DSLOperationsSuite
  extends FunSuite with TensorFramesTestSparkContext with GraphScoping with Logging {
  lazy val sql = sqlContext

  testGraph("Reduce") {
    val x = constant(Seq(1.0, 1.0)) named "x"
    val out = reduce_sum(x, Seq(0)) named "out"
    assert(out.dims === Nil)
    val df = sql.createDataFrame(Seq(Tuple1(1))).toDF("a")
    val df2 = df.mapRows(out).select("a", "out")
    assert(df2.collect() === Array(Row(1, 2.0)))
  }

  testGraph("Constant") {
    val x = constant(1.0) named "x"
    val df = sql.createDataFrame(Seq(Tuple1(1))).toDF("a")
    val df2 = df.mapRows(x).select("a", "x")
    assert(df2.collect() === Array(Row(1, 1.0)))
  }

  testGraph("Map over multiple rows") {
    val df = make1(Seq(1.0, 2.0), "x")
    val x = placeholder[Double](Unknown) named "x"
    val y = identity(x) named "y"
    val z = x + x named "z"
    val df2 = df.mapBlocks(y, z).select("x", "y", "z")
    assert(df2.collect() === Array(Row(1.0, 1.0, 2.0), Row(2.0, 2.0, 4.0)))
  }

  testGraph("Implicit conversions of scalars") {
    // No broadcasting for now.
    val x = constant(1.0)
    val y = 3.0 + x
  }

  testGraph("Implicit conversions of scalars with broadcast") {
    // No broadcasting for now.
    val x = constant(Seq(1.0, 2.0))
    val y = 3.0 + x
  }

  testGraph("Automatic row extraction") {
    val df = make1(Seq(1.0, 2.0), "a")
    val a = df.row("a")
    val b = a + 2.0 named "b"
    println(s"*****\n$a")
    println(s"*****\n$b")
    val df2 = df.mapRows(b).select("a", "b")
    assert(df2.collect() === Array(
      Row(1.0, 3.0),
      Row(2.0, 4.0)))
  }

  testGraph("Automatic block extraction") {
    val df = make1(Seq(1.0, 2.0), "a")
    val a = df.block("a")
    val b = a + 2.0 named "b"
    val df2 = df.mapBlocks(b).select("a", "b")
    assert(df2.collect() === Array(
      Row(1.0, 3.0),
      Row(2.0, 4.0)))
  }
}