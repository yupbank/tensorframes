package org.tensorframes.dsl

import org.apache.spark.sql.{RelationalGroupedDataset, Row, DataFrame}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{ExperimentalOperations, ShapeDescription, OperationsInterface}
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.test.DslOperations

/**
 * The default implementation of the TensorFrames operations.
 */
object Ops extends OperationsInterface with DslOperations with ExperimentalOperations {
  private val ops = new DebugRowOps

  override def mapRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = ops.mapRows(dataframe, graph, shapeHints)

  override def mapBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = ops.mapBlocks(dataframe, graph, shapeHints)

  override def mapBlocksTrimmed(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = ops.mapBlocksTrimmed(dataframe, graph, shapeHints)

  override def reduceRows(
      dataFrame: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = ops.reduceRows(dataFrame, graph, shapeHints)

  override def reduceBlocks(
      dataFrame: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    ops.reduceBlocks(dataFrame, graph, shapeHints)
  }

  override def aggregate(
      data: RelationalGroupedDataset,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    ops.aggregate(data, graph, shapeHints)
  }

  override def explain(df: DataFrame): String = ops.explain(df)

}
