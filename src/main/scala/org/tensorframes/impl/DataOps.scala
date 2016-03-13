package org.tensorframes.impl

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{NumericType, StructType}
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.tensorframes.{ColumnInformation, Shape}
import org.tensorframes.Shape.DimType

/**
  * Converts data between the C++ runtime of TensorFlow and the Spark runtime.
 *
 * Current (only) implementation exports each row and copies it back into a C++ buffer.
 */
object DataOps extends Logging {

  /**
   * (Slow) implementation that takes data in C++ and puts it back into SQL rows, following
   * the structure provided and merging back all the columns from the input.
    *
    * @param tv
   * @param tf_struct the structure of the block represented in TF
   * @return
   */
  // Note that doing it this way is very inefficient, but columnar implementation should prevent all this
  // data copying in most cases.
  def convertBack(
      tv: jtf.TensorVector,
      tf_struct: StructType,
      input: Array[Row],
      input_struct: StructType): Array[Row] = {
    // The structures should already have been validated.
    // Output has all the TF columns first, and then the other columns
    logDebug(s"convertBack: ${input.length} input rows, tf_struct=$tf_struct")

    val tfiters = for ((t, idx) <- tf_struct.fields.zipWithIndex) yield {
      val info = ColumnInformation(t).stf.getOrElse {
        throw new Exception(s"Missing info in field $t")
      }
      logDebug(s"convertBack: $t $info")
      // Drop the first cell, this is a block.
      getColumn(tv, idx, info.dataType, info.shape.tail, input.length)
    }
    val outputSchema = StructType(tf_struct.fields ++ input_struct.fields)
    val allIters: Array[Iterator[Any]] = {
      if (input_struct.isEmpty) {
        tfiters.map(_.iterator)
      } else {
        val riters = input_struct.indices.map (idx => getColumn(input, idx))
        (tfiters ++ riters).map(_.iterator)
      }
    }
    val res = for (i <- input.indices) yield {
      assert(allIters.forall(_.hasNext))
      val current = allIters.map(_.next())
      new GenericRowWithSchema(current, outputSchema)
    }
    logDebug(s"outputSchema=$outputSchema")
    logDebug(s"res: $res")
    res.toArray
  }

  /**
   * Performs size checks and resolutions, and converts the data from the row format to the C++
   * buffers.
   *
   * @param it
   * @param struct the structure of the block. It should contain all the extra meta-data required by
   *               TensorFrames.
   * @param requestedTFCols: the columns that will be fed into TF
   * @return
   */
  def convert(
      it: Array[Row],
      struct: StructType,
      requestedTFCols: Array[Int]): jtf.StringTensorPairVector = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks.
    logDebug(s"Calling convert on ${it.length} rows with struct: $struct " +
      s"and indices: ${requestedTFCols.toSeq}")
    val fields = requestedTFCols.map(struct.fields(_))
    val converters = fields.map { f =>
      // Extract and check the shape
      val ci = ColumnInformation(f).stf.getOrElse {
        throw new Exception(s"Could not column information for column $f")
      }
      val leadDim = ci.shape.dims.headOption.getOrElse {
        throw new Exception(s"Column $f found to be scalar, but its dimensions should be >= 1")
      } .toInt
      if (leadDim != Shape.Unknown && leadDim != it.length) {
        throw new Exception(s"Lead dimension for column $f (found to be $leadDim)" +
          s" is not compatible with a block of lize ${it.length}")
      }
      SupportedOperations.opsFor(ci.dataType).tfConverter(ci.shape.tail, it.length)
    }
    for (c <- converters) { c.reserve() }

    for (r <- it) {
      for ((c, idx) <- converters.zip(requestedTFCols)) {
        c.append(r, idx)
      }
    }

    val tensors = converters.map(_.tensor())
    val names = requestedTFCols.map(struct(_).name)
    for ((name, t) <- names.zip(tensors)) {
      logDebug(s"convert: $name : ${TensorFlowOps.jtfShape(t.shape())}")
    }
    new jtf.StringTensorPairVector(names, tensors)
  }

  /**
    * Converts a single row at a time.
    *
    * @param r the row to convert
    * @param blockStruct the structure of the block that produced this row
    * @param requestedTFCols the requested columns
    * @return
    */
  def convert(
      r: Row,
      blockStruct: StructType,
      requestedTFCols: Array[Int]): jtf.StringTensorPairVector = {
    // This is a very simple and very inefficient implementation. It should be kept
    // as is for correctness checks. The columnar implementation is meant to be more
    // efficient.
    logDebug(s"Calling convert on one with struct: $blockStruct")
    val elts = requestedTFCols.map { idx =>
      val f = blockStruct.fields(idx)
      // Extract and check the shape
      val ci = ColumnInformation(f).stf.getOrElse {
        throw new Exception(s"Could not column information for column $f")
      }
      assert(ci.shape.numDims >= 1,
        s"Column $f found to be a scala, but its dimensions should be >= 1")
      // Special case: if the cell shape has undefined size in its first argument, we
      // still accept it and attempt to get it from the shape. This is to support rows
      // with different vector sizes. All other dimensions must match, although this
      // could be relaxed in the future as well. It is harder to check.
      val cellShape = {
        val givenCellShape = ci.shape.tail
        if (givenCellShape.dims.headOption == Some(Shape.Unknown)) {
          r.get(idx) match {
            case s: Array[_] =>
              givenCellShape.tail.prepend(s.length.toLong)
            case s: Seq[_] =>
              givenCellShape.tail.prepend(s.length.toLong)
            case _ => givenCellShape
          }
        } else {
          givenCellShape
        }
      }
      assert(!cellShape.hasUnknown,
        s"The computed shape for the cell $idx (field $f) is $cellShape, which has unknowns")

      val conv = SupportedOperations.opsFor(ci.dataType).tfConverter(cellShape, 1)
      conv.reserve()
      conv.append(r, idx)
      f.name -> conv.tensor()
    }
    val names = elts.map(_._1)
    val tensors = elts.map(_._2)
    new jtf.StringTensorPairVector(names, tensors)
  }

  // **** Conversions TF => Spark ***

  private def reshapeIter(it: Iterable[Any], s: List[DimType]): Seq[Any] = {
    // The reshaping is extremely inefficient but the output is easy to read after that.
    s match {
      case Nil => throw new Exception()
      case n :: Nil =>
        val x = it.toArray.toSeq
        assert(x.size == n.toInt, s"$n $x")
        x
      case n :: t =>
        val blockSize = t.product.toInt
        val x = it.grouped(blockSize).map(reshapeIter(_, t)).toArray.toSeq
        assert(x.size == n.toInt, s"$n:$t $it")
        x
    }
  }

  /**
   * Extracts the content of a column as objects amenable to SQL.
    *
    * @param tv
   * @param position
   * @param scalaType the scalar type of the tensor
   * @param cellShape the shape of each cell of data
   * @return
   */
  private def getColumn(
      tv: jtf.TensorVector,
      position: Int,
      scalaType: NumericType,
      cellShape: Shape,
      numRows: Int): Iterable[Any] = {
    val t = tv.get(position)
    logDebug(s"getColumn: shape: ${TensorFlowOps.jtfShape(t.shape())}  " +
      s"cellShape:$cellShape numRows:$numRows")
    logDebug(s"getColumn: got tensor: ${t.DebugString().toString}")
    val rawBuff = t.tensor_data().asBuffer()
    val allDataBuffer = SupportedOperations.opsFor(scalaType).convertBuffer(rawBuff)
    val numData = allDataBuffer.size
    // Infer if necessary the reshaping size.
    val reshapeShape = if (cellShape.hasUnknown) {
      val numUnknowns = cellShape.dims.count(_.toInt == Shape.Unknown)
      require(numUnknowns == 1, s"Shape $cellShape for position $position has too many unknowns")
      val h = cellShape.dims.takeWhile(_ >= 0)
      val t = cellShape.dims.dropWhile(_ >= 0).tail
      val known = (numRows.toLong +: (h ++ t)).product.toInt
      val inferred = numData / known
      assert(known * inferred == numData,
        s"Could not infer missing shape from $numRows rows with shape $cellShape " +
          s"and $numData elements in buffer")
      // Reconstruct the final shape
      val s = Shape(((h :+ inferred.toLong) ++ t).toArray).prepend(numRows)
      logDebug(s"Inferred shape: $cellShape -> ($h, $t) -> $s")
      s
    } else {
      cellShape.prepend(numRows)
    }
    logDebug(s"getColumn: databuffer = $allDataBuffer")
    logDebug(s"getColumn: reshapeShape: $reshapeShape, numData: $numData")
    val res = reshapeIter(allDataBuffer, reshapeShape.dims.toList)
    logDebug(s"getColumn: reshaped = $res")
    res
  }

  private def getColumn(rows: Iterable[Row], position: Int): Iterable[Any] = {
    rows.map(_.get(position))
  }
}
