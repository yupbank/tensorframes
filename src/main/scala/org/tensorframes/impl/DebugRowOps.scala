package org.tensorframes.impl

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{GroupedData, DataFrame, Row}
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.tensorflow.framework.GraphDef
import org.tensorframes._
import org.tensorframes.test.DslOperations

/**
  * All the schema transformations that are done by the basic TF operations.
  *
  * These methods describe the schema transforms performed on a DataFrame. They include
  * all the validation steps that should be performed before passing data to TensorFlow.
  *
  * After calling these methods, the implementation can assume the schemas are valid and complete enough.
  */
// Implementation is separated for python accessors
private[impl] trait SchemaTransforms extends Logging {
  def get[A](x: Option[A], msg: String) = x.getOrElse {
    throw new Exception(msg)
  }

  def check(b: Boolean, msg: String): Unit = if (! b) {
    throw new Exception(msg)
  }

  /**
    *
    * @param schema the schema of the dataframe
    * @param graph the graph
    * @param shapeHints the shape hints obtained for this graph
    * @return a pair of inputSchema, outputSchema. The schemas are for the blocks.
    */
  def reduceBlocksSchema(
      schema: StructType,
      graph: GraphDef,
      shapeHints: ShapeDescription): (StructType, StructType) = {
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val fieldsByName = schema.fields.map(f => f.name -> f).toMap
    val fieldNameList = fieldsByName.keySet.toSeq.sorted.mkString(", ")
    val outputNameList = summary.filter(_._2.isOutput).keySet.toSeq.sorted.mkString(", ")
    val suffix = "_input"

    // Initial check: all the fields are here:
    val outputs = summary.filter(_._2.isOutput)
    // Check that there are precisely as many outputs as columns:
    val extraOutputs = (outputs.keySet -- fieldsByName.keySet).toSeq.sorted
    check(extraOutputs.isEmpty, {
      val extra = extraOutputs.mkString(", ")
      s"Some extra outputs were found in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList" })

    val missingOutputs = (fieldsByName.keySet -- outputs.keySet).toSeq.sorted
    check(missingOutputs.isEmpty, {
      val extra = missingOutputs.mkString(", ")
      s"Some outputs are missing in the reducer: $extra. " +
        s"Dataframe columns: $fieldNameList; Outputs: $outputNameList" })

    // Initial check: the inputs are all there:
    val inputs = summary.filter(_._2.isInput)
    val expectedInputs = fieldsByName.keys.map(_ + suffix).toSet
    val extraInputs = (inputs.keySet -- expectedInputs).toSeq.sorted
    logDebug(s"reduceRows: expectedInputs=$expectedInputs")
    check(extraInputs.isEmpty,
      s"Extra graph inputs have been found: ${extraInputs.mkString(", ")}. Dataframe columns: $fieldNameList")

    val missingInputs = (expectedInputs -- inputs.keySet).toSeq.sorted
    check(missingInputs.isEmpty,
      s"Some inputs are missing in the graph: ${missingInputs.mkString(", ")}. Dataframe columns: $fieldNameList")

    // Check that all the fields are here. Each field schema is the column schema.
    val fields = for (f <- fieldsByName.values) yield {
      val ci = ColumnInformation(f)
      val stf = get(ci.stf,
        s"Data column '${f.name}' has not been analyzed yet, cannot run TF on this dataframe")
      // Check that the output is compatible (its presence has already been checked.
      val out = summary(f.name)
      check(out.isOutput, s"Graph node '${out.name}' should be an output")

      check(stf.dataType == out.scalarType, s"Output '${f.name}' has type ${out.scalarType} but the column type " +
        s"is ${stf.dataType}")

      // Take the tail, we only compare cells
      val cellShape = stf.shape.tail
      check(out.shape.checkMorePreciseThan(cellShape),
        s"Output '${f.name}' has shape ${out.shape}, not compatible with the shapes" +
          s"of field elements $cellShape")

      val inputName = f.name + suffix
      val in = get(summary.get(inputName),
        s"The graph needs to have a placeholder input called $inputName.")
      assert(in.isPlaceholder, s"Node $inputName should be a placeholder")
      assert(in.isInput, s"Node $inputName should be an input")
      check(stf.shape.checkMorePreciseThan(in.shape),
        s"The data column '${f.name}' has shape ${stf.shape}, not compatible with shape" +
          s" ${in.shape} requested by the TF graph")
      check(stf.dataType == in.scalarType,
        s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data" +
          s" type of the column (${in.scalarType})")
      ColumnInformation(f, stf).merged
    }
    val outputSchema = StructType(fields.toArray)
    // The input schema is simply the block schema, with a different name for the variables.
    val inputSchema = StructType(schema.map { f =>
      f.copy(name = f.name + "_input")
    })
    inputSchema -> outputSchema
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
}

object SchemaTransforms extends SchemaTransforms

/**
 * A simple and slow implementation of the basic operations that maximizes correctness of
 * implementation and works with older versions of Spark (based on RDDs).
 */
class DebugRowOps
  extends OperationsInterface
    with ExperimentalOperations with DslOperations with PythonInterface with Logging {

  import SchemaTransforms._

  override def mapBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    val sc = dataframe.sqlContext.sparkContext
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)
    val fieldsByName = dataframe.schema.fields.map(f => f.name -> f).toMap
    val cols = dataframe.schema.fieldNames.mkString(", ")
    // The input schema, after validation with TF that it contains all the spark TF info.
    val inputSchema: StructType = {

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
            s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type of the column (${in.scalarType})")
        }
        // The input has to be either a constant or a placeholder
        if (! in.isPlaceholder) {
          throw new Exception(
            s"Invalid type for input node ${in.name}. It has to be a placeholder")
        }
      }
      dataframe.schema
    }

    // The output schema from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        if (fieldsByName.contains(out.name)) {
          throw new Exception(s"TF graph has an output node called '${out.name}'," +
            s" but this column already exists. Input columns: ${cols}")
        }
        ColumnInformation.structField(out.name, out.scalarType, out.shape)
      }
      StructType(fields.toArray)
    }
    // The column indices requested by TF
    val requestedTFInput: Array[Int] = {
      val colIdxs = dataframe.schema.fieldNames.zipWithIndex.toMap
      inputSchema.fieldNames.map { name => colIdxs(name) }
    }
    // Full output schema, including data being passed through and validated for duplicates.
    // The first columns are the TF columns, followed by all the other columns.
    val outputSchema: StructType = {
      StructType(outputTFSchema ++ dataframe.schema.fields)
    }

    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      DebugRowOpsImpl.performMap(
        it.toArray,
        inputSchema,
        requestedTFInput,
        gProto.value,
        outputTFSchema).toIterator
    }
    dataframe.sqlContext.createDataFrame(transformRdd, outputSchema)
  }

  override def mapRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = {
    val sc = dataframe.sqlContext.sparkContext
    val summary = TensorFlowOps.analyzeGraph(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)
    val fieldsByName = dataframe.schema.fields.map(f => f.name -> f).toMap
    val cols = dataframe.schema.fieldNames.mkString(", ")

    inputs.values.foreach { in =>
      val f = get(fieldsByName.get(in.name),
        s"Graph input ${in.name} found, but no column to match it. Dataframe columns: $cols")

      val stf = get(ColumnInformation(f).stf,
        s"Data column ${f.name} has not been analyzed yet, cannot run TF on this dataframe")

      val cellShape = stf.shape.tail
      // No check for unknowns: we allow unknowns in the first dimension of the cell shape.
      check(cellShape.checkMorePreciseThan(in.shape),
        s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
          s" ${in.shape} requested by the TF graph")

      check(stf.dataType == in.scalarType,
        s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type " +
          s"of the column (${in.scalarType})")

      check(in.isPlaceholder,
        s"Invalid type for input node ${in.name}. It has to be a placeholder")
    }

    // The output schema of the block from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        check(! fieldsByName.contains(out.name),
          s"TF graph has an output node called '${out.name}'," +
            s" but this column already exists. Input columns: ${cols}")
        // The shapes we get in each output node are the shape of the cells of each column, not the
        // shape of the column. Add Unknown since we do not know the exact length of the block.
        val blockShape = out.shape.prepend(Shape.Unknown)
        ColumnInformation.structField(out.name, out.scalarType, blockShape)
      }
      StructType(fields.toArray)
    }

    // The column indices requested by TF
    val requestedTFInput: Array[Int] = {
      val colIdxs = dataframe.schema.fieldNames.zipWithIndex.toMap
      dataframe.schema.fieldNames.map { name => colIdxs(name) }
    }
    // Full output schema, including data being passed through and validated for duplicates.
    // The first columns are the TF columns, followed by all the other columns.
    val outputSchema: StructType = {
      StructType(outputTFSchema ++ dataframe.schema.fields)
    }

    val schema = dataframe.schema // Classic rookie mistake...
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      DebugRowOpsImpl.performMapRows(
        it.toArray,
        schema,
        requestedTFInput,
        gProto.value,
        outputTFSchema).toIterator
    }
    dataframe.sqlContext.createDataFrame(transformRdd, outputSchema)
  }

  override def reduceRows(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    val sc = dataframe.sqlContext.sparkContext
    // Most of this function is various sanity checks on the graph and on the dataframe.
    reduceRowsSchema(dataframe.schema, graph, shapeHints)
    val schema = dataframe.schema
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      val row = DebugRowOpsImpl.performReducePairwise(
        it.toArray, schema, gProto.value
      )
      Array(row).iterator
    }
    transformRdd.reduce(DebugRowOpsImpl.reducePair(schema, gProto))
  }

  override def reduceBlocks(
      dataframe: DataFrame,
      graph: GraphDef,
      shapeHints: ShapeDescription): Row = {
    val sc = dataframe.sqlContext.sparkContext
    val schema = dataframe.schema
    val (inputSchema, outputSchema) = reduceBlocksSchema(schema, graph, shapeHints)
    val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
    // It first reduces each block, and then performs pair-wise reduction.
    val transformRdd = dataframe.rdd.mapPartitions { it =>
      val row = DebugRowOpsImpl.performReduceBlock(
        it.toArray, inputSchema, outputSchema, gProto.value)
      Array(row).iterator
    }
    transformRdd.reduce(DebugRowOpsImpl.reducePairBlock(inputSchema, outputSchema, gProto))
  }

  override def explain(df: DataFrame): String = {
    val d = explainDetailed(df)

    val builder = new StringBuilder
    builder.append("root\n")
    val prefix = " |"
    d.cols.foreach { col =>
      val f = col.field
      builder.append(s"$prefix-- ${f.name}: ${f.dataType.typeName} (nullable = ${f.nullable})")
      val stf = col.stf.map { s =>
        s" ${s.dataType}${s.shape}"
      }   .getOrElse("No tensor info")
      builder.append(stf)
      builder.append("\n")
    }
    builder.toString()
  }

  override def aggregate(
      data: GroupedData,
      graph: GraphDef,
      shapeHints: ShapeDescription): DataFrame = ???
}

object DebugRowOpsImpl extends Logging {

  private[impl] def reducePair(
      schema: StructType,
      gbc: Broadcast[Array[Byte]]): (Row, Row) => Row = {
    def f(row1: Row, row2: Row): Row = {
      performReducePairwise(Array(row1, row2), schema, gbc.value)
    }
    f
  }

  def reducePairBlock(
      inputSchema: StructType,
      outputSchema: StructType,
      gbc: Broadcast[Array[Byte]]): (Row, Row) => Row = {
    def f(row1: Row, row2: Row): Row = {
      performReduceBlock(Array(row1, row2), inputSchema, outputSchema, gbc.value)
    }
    f
  }
  /**
   * Performs the data transform. All the data has been verified at this point.
   * 
   * The output format is: all the columns coming out of TF first (their exact content being given
   * by tfOutputSchema), and then all the input columns.
   *
   * @param input the row of input data
   * @param inputSchema the schema of the block of inputs
   * @param inputTFCols the columns of input data that will be converted to TF data and fed into the
   *                    TF session
   * @param graphDef the graph definition that contains the description of the TF graph
   * @param tfOutputSchema the (expected) output schema of the block of outputs
   * @return the array of rows that combine outputs from the TF graph with all the other data
   *         columns.
   */
  def performMap(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[Int],
      graphDef: Array[Byte],
      tfOutputSchema: StructType): Array[Row] = {
    logDebug(s"performMap: inputSchema=$inputSchema, tfschema=$tfOutputSchema, ${input.length} rows")
    val stpv = DataOps.convert(input, inputSchema, inputTFCols)
    val g = TensorFlowOps.readGraph(graphDef)
    TensorFlowOps.withSession { session =>
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)

      val outputs = new jtf.TensorVector()
      val requested = TensorFlowOps.stringVector(tfOutputSchema.map(_.name))
      val skipped = new jtf.StringVector()
      val s3 = session.Run(stpv, requested, skipped, outputs)
      assert(s3.ok(), s3.error_message().getString)
      DataOps.convertBack(outputs, tfOutputSchema, input, inputSchema)
    }
  }

  // For testing only:
  def performMap(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[Int],
      graphDef: GraphDef,
      outputSchema: StructType): Array[Row] = {
    performMap(input, inputSchema, inputTFCols,
      TensorFlowOps.graphSerial(graphDef), outputSchema)
  }

  def performMapRows(
      input: Array[Row],
      inputSchema: StructType,
      inputTFCols: Array[Int],
      graphDef: Array[Byte],
      tfOutputSchema: StructType): Array[Row] = {
    // We read the graph once, and within the same session we run each row after the other.
    val g = TensorFlowOps.readGraph(graphDef)
    TensorFlowOps.withSession { session =>
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)
      val requested = TensorFlowOps.stringVector(tfOutputSchema.map(_.name))

      input.map { row =>
        val stpv = DataOps.convert(row, inputSchema, inputTFCols)
        val outputs = new jtf.TensorVector()
        val skipped = new jtf.StringVector()
        val s3 = session.Run(stpv, requested, skipped, outputs)
        assert(s3.ok(), s3.error_message().getString)
        DataOps.convertBack(outputs, tfOutputSchema, Array(row), inputSchema) match {
          case Array(r) => r
          case x =>
            throw new Exception(s"Should have received one row, received ${x.toList}")
        }
      }
    }
  }

  /**
   * Performs a reduce operation on the set of columns.
   *
   * This performs a fix-point operation around the given schema:
   *  - all the columns are transformed
   *  - for each output X there is a given X_input input. The schema is assumed to be the same with
   *  an extra block dimension.
   *
   * @param input the input of rows.
   * @param schema the output schema.
   * @param graphDef the definition of the graph
   * @return the combined row after transform
   */
  def performReduceBlock(
      input: Array[Row],
      inputSchema: StructType,
      schema: StructType,
      graphDef: Array[Byte]): Row = {
    logDebug(s"performReduceBlock: schema=$schema inputSchema=$inputSchema with ${input.length} rows")
    // The input schema and the actual data representation depend on the block operation.

    val stpv = DataOps.convert(input, inputSchema, schema.indices.toArray)
    val g = TensorFlowOps.readGraph(graphDef)
    TensorFlowOps.withSession { session =>
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)

      val outputs = new jtf.TensorVector()
      val requested = TensorFlowOps.stringVector(schema.map(_.name))
      val skipped = new jtf.StringVector()
      val s3 = session.Run(stpv, requested, skipped, outputs)
      assert(s3.ok(), s3.error_message().getString)
      val emptyRows = Array.fill(1)(emptyRow)
      DataOps.convertBack(outputs, schema, emptyRows, emptySchema).head
    }
  }

  private val emptySchema = StructType(Seq.empty)
  private val emptyRow: Row = new GenericRowWithSchema(Array.empty[Any], emptySchema)

  /**
   * Performs a reduce operation on a set of columns by performing pair-wise reductions.
   *
   * @param input
   * @param schema
   * @param graphDef
   * @return
   */
  def performReducePairwise(input: Array[Row], schema: StructType, graphDef: Array[Byte]): Row = {
    require(input.length > 0, "Cannot provide empty input")
    // If there is a single row, no need to perform operations.
    if (input.length == 1) {
      return input.head
    }
    val inputSchema = {
      val f1 = schema.fields.map(f => f.copy(name = s"${f.name}_1"))
      val f2 = schema.fields.map(f => f.copy(name = s"${f.name}_2"))
      StructType(f1 ++ f2)
    }
    // For efficiency, this tries to reuse the session.
    val tfInput: Array[Row] = input.tail
    var result: Row = input.head
    val g = TensorFlowOps.readGraph(graphDef)
    TensorFlowOps.withSession { session =>
      val s1 = session.Extend(g)
      assert(s1.ok(), s1.error_message().getString)
      for (row <- tfInput) {
        val values = (result.toSeq ++ row.toSeq).toArray
        val r: Row = new GenericRowWithSchema(values, inputSchema)
        val thisInput = Array(r)
        val stpv = DataOps.convert(thisInput, inputSchema, inputSchema.fields.indices.toArray)
        val outputs = new jtf.TensorVector()
        val requested = TensorFlowOps.stringVector(schema.map(_.name))
        val skipped = new jtf.StringVector()
        val s3 = session.Run(stpv, requested, skipped, outputs)
        assert(s3.ok(), s3.error_message().getString)
        // Fill in with an empty row, because we are not passing the rest of of the data.
        val emptyRows = Array.fill(1)(emptyRow)
        result = DataOps.convertBack(outputs, schema, emptyRows, emptySchema) match {
          case Array(finalRow) => finalRow
          case x => throw new Exception(s"Should be one row, at ${x.toList}")
        }
      }
    }
    result
  }
}

object DebugRowOps extends DebugRowOps
