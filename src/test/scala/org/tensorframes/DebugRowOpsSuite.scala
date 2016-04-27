package org.tensorframes

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.scalatest.FunSuite
import org.tensorframes.impl.DebugRowOpsImpl
// TODO: replace
import org.tensorframes.test.dsl._
import org.tensorframes.test.DslOperations

class DebugRowOpsSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging {
  lazy val sql = sqlContext
  import ColumnInformation.structField
  import Shape.Unknown

  test("Simple identity") {
    val rows = Array(Row(1.0))
    val input = StructType(Array(structField("x", DoubleType, Shape(Unknown))))
    val p2 = placeholder(DoubleType, Shape(1)) named "x"
    val out = op_id(p2) named "y"
    val outputSchema = StructType(Array(structField("y", DoubleType, Shape(Unknown))))
    val (g, _) = DslOperations.analyzeGraph(out)
    logDebug(g.toString)
    val res = DebugRowOpsImpl.performMap(rows, input, Array(0), g, outputSchema)
    assert(res === Array(Row(1.0, 1.0)))
  }

  test("Simple add") {
    val rows = Array(Row(1.0))
    val input = StructType(Array(structField("x", DoubleType, Shape(Unknown))))
    val p2 = placeholder(DoubleType, Shape(1)) named "x"
    val out = p2 + p2 named "y"
    val outputSchema = StructType(Array(structField("y", DoubleType, Shape(Unknown))))
    val (g, _) = DslOperations.analyzeGraph(out)
    logDebug(g.toString)
    val res = DebugRowOpsImpl.performMap(rows, input, Array(0), g, outputSchema)
    assert(res === Array(Row(2.0, 1.0)))
  }

}
