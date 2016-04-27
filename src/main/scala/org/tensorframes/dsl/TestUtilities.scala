package org.tensorframes.dsl

import java.nio.file.{Paths, Files}

import org.tensorflow.framework.GraphDef

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

}
