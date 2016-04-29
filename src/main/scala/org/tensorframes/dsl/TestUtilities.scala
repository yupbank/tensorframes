package org.tensorframes.dsl

import java.nio.file.{Paths, Files}

import org.tensorflow.framework.GraphDef
import org.tensorframes.ShapeDescription
import org.tensorframes.impl.{TensorFlowOps, GraphNodeSummary}

/**
 * Some utilities for running tests.
 */
private[tensorframes] object TestUtilities {

  def buildGraph(node: Operation, nodes: Operation*): GraphDef = {
    DslImpl.buildGraph(Seq(node) ++ nodes)
  }

  def loadGraph(file: String): GraphDef = {
    val byteArray = Files.readAllBytes(Paths.get(file))
    GraphDef.newBuilder().mergeFrom(byteArray).build()
  }

  private def extraInfo(fetches: Seq[Node]): ShapeDescription = {
    ShapeDescription(
      fetches.map(n => n.name -> n.shape).toMap,
      fetches.map(_.name))
  }


  def analyzeGraph(nodes: Operation*): (GraphDef, Seq[GraphNodeSummary]) = {
    val g = buildGraph(nodes.head, nodes.tail: _*)
    g -> TensorFlowOps.analyzeGraph(g, extraInfo(nodes))
  }

}
