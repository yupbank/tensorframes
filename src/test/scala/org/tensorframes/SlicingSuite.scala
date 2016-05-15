package org.tensorframes

import org.scalatest.FunSuite
import org.tensorframes.dsl.GraphScoping
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType}


class SlicingSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging with GraphScoping {
  lazy val sql = sqlContext

  import Shape.Unknown

  val ops = new DebugRowOps

  test("2D - 1") {
    val df = make1(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)), "x")
    val x = df.block("x")
//    val y =
  }
}