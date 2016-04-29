package org.tensorframes.dsl

import org.apache.spark.sql.{GroupedData, Row, DataFrame}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{OperationsInterface, ShapeDescription, dsl}
import org.tensorframes.impl.DebugRowOps

/**
 * Implicit transforms to help with the construction of TensorFrames manipulations.
 */
trait DFImplicits {
  protected def ops: OperationsInterface

  implicit class RichDataFrame(df: DataFrame) {
    def mapRows(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapRows(df, graph, shapeHints)
    }

    def mapRows(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(seq)
      mapRows(g, Node.hints(seq))
    }

    def mapBlocks(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapBlocks(df, graph, shapeHints)
    }

    def mapBlocks(o0: Operation, os: Operation*): DataFrame = {
      val seq = Seq(o0) ++ os
      val g = DslImpl.buildGraph(Seq(o0) ++ os)
      mapBlocks(g, Node.hints(seq))
    }

    def reduceRows(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceRows(df, graph, shapeHints)
    }

    def reduceBlocks(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceBlocks(df, graph, shapeHints)
    }

    def explainTensors: String = ops.explain(df)

    // TODO: add analysis
  }

  implicit class RichGroupedData(dg: GroupedData) {
    def aggregate(graphDef: GraphDef, shapeDescription: ShapeDescription): DataFrame = {
      ops.aggregate(dg, graphDef, shapeDescription)
    }
  }

}

object Implicits extends DFImplicits with DefaultConversions {
  override def ops = Ops

}
