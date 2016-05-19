package org.tensorframes.catalyst

import org.apache.spark.{LoggingWrapper => Logging}
import org.apache.spark.sql._
import org.tensorflow.framework.GraphDef
import org.tensorframes.impl.{TensorFlowOps, DebugRowOps}
import org.tensorframes.impl.SchemaTransforms._
import org.tensorframes.{ShapeDescription, OperationsInterface}

/**
 * Optimized implementation of the TensorFrames operation that hooks directly into the catalyst
 * compiler.
 */
object CatalystOperations extends OperationsInterface with Logging {

  override def mapRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    DebugRowOps.mapRows(dataframe, graph, shapeHints)
  }

  override def mapBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    DebugRowOps.mapBlocks(dataframe, graph, shapeHints)
  }

  override def mapBlocksTrimmed(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    DebugRowOps.mapBlocksTrimmed(dataframe, graph, shapeHints)
  }

  override def reduceRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    DebugRowOps.reduceRows(dataframe, graph, shapeHints)
  }

  override def reduceBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    DebugRowOps.reduceBlocks(dataframe, graph, shapeHints)
  }

  override def aggregate(
      data: RelationalGroupedDataset,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    DebugRowOps.aggregate(data, graph, shapeHints)
  }

  override def explain(df: DataFrame): String = {
    DebugRowOps.explain(df)
  }

  private def mapBlocks(
    dataframe: DataFrame,
    graph: GraphDef,
    shapeHints: ShapeDescription,
    appendInput: Boolean): DataFrame = {
    val sc = dataframe.sqlContext.sparkContext
    val transform = mapBlocksSchema(dataframe.schema, graph, shapeHints, appendInput)

    logDebug(s"mapBlocks: TF input schema = ${transform.inputSchema}," +
      s" complete output schema = ${transform.outputSchema}")

    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val child = TFHooks.logicalPlan(dataframe)
    val plan = TestMapBlockPlan(child, gProto, transform)
    TFHooks.ofRows(dataframe.sparkSession, plan)
  }

}

