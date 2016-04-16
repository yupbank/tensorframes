package org.tensorframes.dsl

import org.apache.spark.sql.{GroupedData, Row, DataFrame}
import org.tensorflow.framework.GraphDef
import org.tensorframes.ShapeDescription
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.dsl

/**
 * Implicit transforms to help with the construction of TensorFrames manipulations.
 */
object Implicits {
  import dsl._

  private val ops = new DebugRowOps


  implicit class RichDataFrame(df: DataFrame) {
    def mapRows(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapRows(df, graph, shapeHints)
    }

    def mapBlocks(graph: GraphDef, shapeHints: ShapeDescription): DataFrame = {
      ops.mapBlocks(df, graph, shapeHints)
    }

    def reduceRows(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceRows(df, graph, shapeHints)
    }

    def reduceBlocks(graph: GraphDef, shapeHints: ShapeDescription): Row = {
      ops.reduceBlocks(df, graph, shapeHints)
    }

    def explainTensors: String = ops.explain(df)
  }

  implicit class RichGroupedData(dg: GroupedData) {
    def aggregate(graphDef: GraphDef, shapeDescription: ShapeDescription): DataFrame = {
      ops.aggregate(dg, graphDef, shapeDescription)
    }
  }

  implicit class RichDouble(x: Double) {
    def +(n: Operation): Operation = {
      constant(x) + n
    }
  }

}
