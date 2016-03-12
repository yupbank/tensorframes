package org.tensorframes

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.scalatest.FunSuite
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.test.dsl._

class DSLOperationsSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging {
  lazy val sql = sqlContext
  import Shape.Unknown

  val ops = new DebugRowOps

  test("Reduce") {
    val x = constant(Seq(1.0, 1.0)) named "x"
    val out = reduce_sum(x, Seq(0)) named "out"
    assert(out.shape === Shape.empty)
    val df = sql.createDataFrame(Seq(Tuple1(1))).toDF("a")
    val df2 = ops.mapRows(df, out).select("a", "out")
    assert(df2.collect() === Array(Row(1, 2.0)))
  }

  test("Constant") {
    val x = constant(1.0) named "x"
    val df = sql.createDataFrame(Seq(Tuple1(1))).toDF("a")
    val df2 = ops.mapRows(df, x).select("a", "x")
    assert(df2.collect() === Array(Row(1, 1.0)))
  }

}