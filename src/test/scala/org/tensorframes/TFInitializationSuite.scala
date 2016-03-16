package org.tensorframes

import org.apache.spark.Logging
import org.apache.spark.sql.types.DoubleType
import org.scalatest.FunSuite
import org.tensorframes.impl.TensorFlowOps
import org.tensorframes.test.dsl


class TFInitializationSuite extends FunSuite with Logging {

  import dsl._

  test("Simple loading of a placeholder") {
    val p1 = placeholder(DoubleType, Shape(1)) named "p1"
    val p2 = placeholder(DoubleType, Shape(1)) named "p2"
    val a = p1 + p2 named "a"
    val g = buildGraph(a)
    logDebug(g.toString)
    TensorFlowOps.analyzeGraph(g, ShapeDescription(Map("a"->Shape(1)), Seq("a")))
  }

  // Requires support for floats
  // TODO(tjh) activate when floats are supported
  ignore("Loading an existing graph") {
    val g2 = load("/home/tensorframes/src/test/resources/graph2.pb")
    logDebug(g2.toString)
    TensorFlowOps.analyzeGraph(g2)
  }

  test("Graph with identity") {
    val p2 = placeholder(DoubleType, Shape(1)) named "x"
    val out = op_id(p2) named "y"
    val g = buildGraph(out)
    TensorFlowOps.analyzeGraph(g, ShapeDescription(Map("y"->Shape(1)), Seq("y")))
  }
}