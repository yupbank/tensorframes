package org.tensorframes.impl

import org.apache.spark.{LoggingWrapper => Logging}
import org.apache.spark.sql.types.{StructField, StructType}
import org.tensorflow.framework.GraphDef
import org.tensorframes.{Shape, ColumnInformation, ShapeDescription}


/**
 * The different schemas required for the block reduction.
 *
 * Here is the order (block reduction in a map phase followed by pair-wise aggregation in a
 * reduce phase):
 *  mapInput -> output -> reduceInput -> output
 *
 * Call 'x' a variable required by the transform, and 'y' an extra column
 *
 * @param mapInput the schema of the column block. Contains 'x_input' and 'y'
 * @param mapTFCols the indexes of the columns required by the transform
 * @param output contains 'x' only
 * @param reduceInput contains 'x_input' only ('y' column has been dropped in the map phase).
 */
case class ReduceBlockSchema(
    mapInput: StructType,
    mapTFCols: List[Int],
    output: StructType,
    reduceInput: StructType) extends Serializable

/**
 * The schemas required by the block mapping.
 *
 * Trimming can be inferred from the condition outputTFSchema == outputSchema
 *
 * @param inputSchema the schema of the input dataframe
 * @param mapTFCols the list of indexes in the previous schema of the columns required by the TF
 *                  mapping.
 * @param outputTFSchema the schema of the columns created by TF
 * @param outputSchema the complete schema of the final dataframe.
 */
case class MapBlocksSchema(
    inputSchema: StructType,
    mapTFCols: Array[Int],
    outputTFSchema: StructType,
    outputSchema: StructType) extends Serializable

/**
 * All the schema transformations that are done by the basic TF operations.
 *
 * These methods describe the schema transforms performed on a DataFrame. They include
 * all the validation steps that should be performed before passing data to TensorFlow.
 *
 * After calling these methods, the implementation can assume the schemas are valid and complete
 * enough.
 */
// Implementation is separated for python accessors
// TODO: these methods are pretty complicated, add more documentation!
private[impl] trait SchemaTransforms extends Logging {
  def get[A](x: Option[A], msg: String) = x.getOrElse {
    throw new Exception(msg)
  }

  def check(b: Boolean, msg: String): Unit = if (! b) {
    throw new Exception(msg)
  }


  /**
   * Validates and computes the transformation schema under reducing.
   *
   * A graph may access a subset of the rows. All the schemas are for blocks of data.
   *
   * For each output x of the graph, there must be:
   *  - a placeholder called x_input with one extra dimension (left unknown)
   *  - a corresponding column labeled 'x' with the same row structure as the output
   *
   * @param schema the schema of the dataframe
   * @param graph the graph
   * @param shapeHints the shape hints obtained for this graph
   * @return a triplet containing the input block schema, the output block schema, and the
   *         requested inputs, which may be a subset of the input.
   */
  // TODO(tjh) all the inputs and outputs are created by hashmaps, which makes their order not
  // deterministic. Change that.
  def reduceBlocksSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription): ReduceBlockSchema = {
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val fieldNameList = fieldsByName.keySet.toSeq.sorted.mkString(", ")
    val outputNameList = summary.filter(_._2.isOutput).keySet.toSeq.sorted.mkString(", ")
    val suffix = "_input"

    // Initial check: all the fields are here:
    val outputs = summary.filter(_._2.isOutput)

    // Check that the outputs of the graph are a subset of the columns of the dataframe.
    val missingColInputs = (outputs.keySet -- fieldsByName.keySet).toSeq.sorted
    check(missingColInputs.isEmpty, {
      val missing = missingColInputs.mkString(", ")
      s"Based on the TF graph, some inputs are missing: $missing. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList" })

    // Initial check: the inputs are all there, and they are the only ones.
    val inputs = summary.filter(_._2.isInput)
    val expectedInputs = outputs.keySet.map(_ + suffix)
    val extraInputs = (inputs.keySet -- expectedInputs).toSeq.sorted
    logDebug(s"reduceRows: expectedInputs=$expectedInputs")
    check(extraInputs.isEmpty,
      s"Extra graph inputs have been found: ${extraInputs.mkString(", ")}. " +
        s"Dataframe columns: $fieldNameList")

    val missingInputs = (expectedInputs -- inputs.keySet).toSeq.sorted
    check(missingInputs.isEmpty,
      s"Some inputs are missing in the graph: ${missingInputs.mkString(", ")}. " +
        s"Dataframe columns: $fieldNameList")

    // Check that for each output, the field is present with the right schema.
    // WARNING: keeping the order of the fields is important -> do not iterate over the outputs.
    val fields = schema.filter(f => outputs.contains(f.name)).map { f =>
      val ci = ColumnInformation(f)
      val stf = get(ci.stf,
        s"Data column '${f.name}' has not been analyzed yet, cannot run TF on this dataframe")
      // Check that the output is compatible (its presence has already been checked.
      val out = summary(f.name)
      check(out.isOutput, s"Graph node '${out.name}' should be an output")

      check(stf.dataType == out.scalarType, s"Output '${f.name}' has type ${out.scalarType}" +
        s" but the column type " +
        s"is ${stf.dataType}")

      // Take the tail, we only compare cells
      val cellShape = stf.shape.tail
      check(out.shape.checkMorePreciseThan(cellShape),
        s"Output '${f.name}' has shape ${out.shape}, not compatible with the shape " +
          s"of field elements $cellShape")
      // The input block may be too precise with respect to the lead dimension (number of rows),
      // which is usually incorrect when pairwise reductions are performed.
      // Always assume they are unknown for now.
      val shape = cellShape.prepend(Shape.Unknown)
      val inputStf = stf.copy(shape = shape)

      val inputName = f.name + suffix
      val in = get(summary.get(inputName),
        s"The graph needs to have a placeholder input called $inputName.")
      assert(in.isPlaceholder, s"Node $inputName should be a placeholder")
      assert(in.isInput, s"Node $inputName should be an input")
      check(inputStf.shape.checkMorePreciseThan(in.shape),
        s"The data column '${f.name}' has shape ${inputStf.shape}, not compatible with shape" +
          s" ${in.shape} requested by the TF graph")
      check(inputStf.dataType == in.scalarType,
        s"The type of node '${in.name}' (${inputStf.dataType}) is not compatible with the data" +
          s" type of the column (${in.scalarType})")
      val m = ColumnInformation(f, inputStf).merged
      logDebug(s">>> $m -> ${ColumnInformation(m).stf}")
      m
    }
    val outputSchema = StructType(fields.toArray)
    // The input schema is simply the block schema, with a different name for the variables.
    // We still pass all the variables because the filtering is done on the indices selected.
    val inputSchema = StructType(schema.map { f =>
      if (outputs.contains(f.name)) {
        widenLeadDim(f.copy(name = f.name + "_input"))
      } else { f }
    })
    val inputReduceSchema = StructType(schema
      .filter(f => outputs.contains(f.name))
      .map(f => widenLeadDim(f.copy(name=f.name + "_input"))))
    val requestedIndexes = schema.zipWithIndex
      .filter { case (f, idx) => outputs.contains(f.name)}
      .map(_._2) .toList
    ReduceBlockSchema(inputSchema, requestedIndexes, outputSchema, inputReduceSchema)
  }

  def reduceRowsSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription): StructType = {
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val fieldNameList = fieldsByName.keySet.toSeq.sorted.mkString(", ")
    val outputNameList = summary.filter(_._2.isOutput).keySet.toSeq.sorted.mkString(", ")
    val suffixes = Seq("_1", "_2")

    // Initial check: all the fields are here:
    val outputs = summary.filter(_._2.isOutput)
    // Check that there are precisely as many outputs as columns:
    if ((outputs.keySet -- fieldsByName.keySet).nonEmpty) {
      val extra = (outputs.keySet -- fieldsByName.keySet).toSeq.sorted.mkString(", ")
      val s = s"Some extra outputs were found in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList"
      throw new Exception(s)
    }
    if ((fieldsByName.keySet -- outputs.keySet).nonEmpty) {
      val extra = (fieldsByName.keySet -- outputs.keySet).toSeq.sorted.mkString(", ")
      val s = s"Some outputs are missing in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList"
      throw new Exception(s)
    }

    // Initial check: the inputs are all there:
    val inputs = summary.filter(_._2.isInput)
    val expectedInputs = suffixes.flatMap(suff => fieldsByName.keys.map(_ + suff)).toSet
    logDebug(s"reduceRows: expectedInputs=$expectedInputs")
    if ((inputs.keySet -- expectedInputs).nonEmpty) {
      val extra = (inputs.keySet -- expectedInputs).toSeq.sorted.mkString(", ")
      throw new Exception(
        s"Extra graph inputs have been found: $extra. Dataframe columns: $fieldNameList")
    }
    if ((expectedInputs -- inputs.keySet).nonEmpty) {
      val extra = (expectedInputs -- inputs.keySet).toSeq.sorted.mkString(", ")
      throw new Exception(
        s"Some inputs are missing in th graph: $extra. Dataframe columns: $fieldNameList")
    }

    // Check that all the fields are here
    for {
      f <- fieldsByName.values
      suffix <- suffixes
    } {
      val stf = ColumnInformation(f).stf.getOrElse { throw new Exception(
        s"Data column '${f.name}' has not been analyzed yet, cannot run TF on this dataframe")
      }
      // Check that the output is compatible (its presence has already been checked.
      val out = summary(f.name)
      if (!out.isOutput) {
        throw new Exception(
          s"Graph node '${out.name}' should be an output")
      }
      if (stf.dataType != out.scalarType) {
        val s = s"Output '${f.name}' has type ${out.scalarType} but the column type " +
          s"is ${stf.dataType}"
        throw new Exception(s)
      }
      // Take the tail, we only compare cells
      val cellShape = stf.shape.tail
      if (! out.shape.checkMorePreciseThan(cellShape)) {
        throw new Exception(
          s"Output '${f.name}' has shape ${out.shape}, not compatible with the shapes" +
            s"of field elements ${cellShape}")
      }

      // Check that the 2 inputs are compatible:
      for (suffix <- Seq("_1", "_2")) {
        val inputName = f.name + suffix
        val in = summary.getOrElse(inputName, throw new Exception(
          s"The graph needs to have a placeholder input called $inputName."))
        assert(in.isPlaceholder, s"Node $inputName should be a placeholder")
        assert(in.isInput, s"Node $inputName should be an input")
        if (! cellShape.checkMorePreciseThan(in.shape)) {
          throw new Exception(
            s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
              s" ${in.shape} requested by the TF graph")
        }
        if (stf.dataType != in.scalarType) {
          throw new Exception(
            s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data" +
              s" type of the column (${in.scalarType})")
        }
      }
    }
    // Same schema as output
    schema
  }

  def mapBlocksSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription,
      appendInput: Boolean): MapBlocksSchema = {
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val cols = schema.fieldNames.mkString(", ")

    inputs.values.foreach { in =>
      val f = get(fieldsByName.get(in.name),
        s"Graph input ${in.name} found, but no column to match it. Dataframe columns: $cols")

      val stf = ColumnInformation(f).stf.getOrElse {
        throw new Exception(
          s"Data column ${f.name} has not been analyzed yet, cannot run TF on this dataframe")
      }
      if (! stf.shape.checkMorePreciseThan(in.shape)) {
        throw new Exception(
          s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
            s" ${in.shape} requested by the TF graph")
      }
      // We do not support autocasting for now.
      if (stf.dataType != in.scalarType) {
        throw new Exception(
          s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type " +
            s"of the column (${in.scalarType})")
      }
      // The input has to be either a constant or a placeholder
      if (! in.isPlaceholder) {
        throw new Exception(
          s"Invalid type for input node ${in.name}. It has to be a placeholder")
      }
    }

    // The output schema from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        if (fieldsByName.contains(out.name)) {
          throw new Exception(s"TF graph has an output node called '${out.name}'," +
            s" but this column already exists. Input columns: ${cols}")
        }
        logInfo(s"mapBlocks: out = $out")
        ColumnInformation.structField(out.name, out.scalarType, out.shape)
      }
      StructType(fields.toArray)
    }
    // The column indices requested by TF
    val requestedTFInput: Array[Int] = {
      val colIdxs = schema.fieldNames.zipWithIndex.toMap
      inputs.keys.toArray.map { name => colIdxs(name) }
    }

    // Full output schema, including data being passed through and validated for duplicates.
    // The first columns are the TF columns, followed by all the other columns.
    val outputSchema: StructType = if (appendInput) {
      StructType(outputTFSchema ++ schema.fields)
    } else {
      StructType(outputTFSchema)
    }

    MapBlocksSchema(schema, requestedTFInput, outputTFSchema, outputSchema)
  }

  // Sets the lead column to Unknown
  private def widenLeadDim(f: StructField): StructField = {
    ColumnInformation(f).stf match {
      case Some(ci) if ci.shape.numDims >= 1 =>
        val s = ci.shape.tail.prepend(Shape.Unknown)
        ColumnInformation(f, ci.copy(shape = s)).merged
      case _ => f // Nothing to do
    }
  }
}

object SchemaTransforms extends SchemaTransforms
