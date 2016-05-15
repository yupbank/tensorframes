package org.tensorframes.impl

import java.nio._

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, DoubleType, IntegerType, NumericType}
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.tensorflow.framework.DataType
import org.tensorframes.Shape

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

// All the datatypes supported by TensorFrames
// TODO: all the buffer operations should specify their byte orders:
//  - TensorFlow: ???
//  - jvm: ???
//  - protobuf: ???

/**
 * @param shape the shape of the element in the row (not the overall shape of the block)
 * @param numCells the number of cells that are going to be allocated with the given shape.
 * @tparam T
 */
// TODO PERF: investigate the use of @specialized
private[tensorframes] sealed abstract class TensorConverter[T : TypeTag] (
    val shape: Shape,
    val numCells: Int) extends Logging {
  /**
   * Creates memory space for a given number of units of the given shape.
   *
   * Can only be called once.
   */
  def reserve(): Unit

  def appendRaw(t: T): Unit

  def append(row: Row, position: Int): Unit = {
    require(shape.dims.size <= 1)
    shape.numDims match {
      case 0 =>
        appendRaw(row.getAs[T](position))
      case 1 =>
        // TODO PERF: we know this is going to be a wrapped array -> we should should specialize
        // the calls to array calls
        row.getSeq[T](position) match {
          case wa : mutable.WrappedArray[T @unchecked] =>
            // Faster path
            val arr = wa.toArray
            var idx = 0
            while (idx < arr.length) {
              appendRaw(arr(idx))
              idx += 1
            }
          case seq: Seq[T @unchecked] =>
            // Slow path
            seq.foreach(appendRaw)
        }
      case x =>
        throw new Exception(s"Higher order dimension $x from shape $shape is not supported")
    }
  }

  def tensor(): jtf.Tensor

  protected def byteBuffer(): ByteBuffer

  def toByteArray(): Array[Byte] = {
    val buff = byteBuffer()
    val pos = buff.position()
    buff.rewind()
    val res = Array.fill[Byte](buff.limit())(0)
    buff.get(res, 0, buff.limit())
    buff.position(pos)
    res
  }
}


/**
 * All the operations required to support a given type.
 *
 * It assumes that there is a one-to-one correspondence between SQL types, scala types and TF types.
 * It does not support TF's rich type collection (uint16, float128, etc.). These have to be handled
 * internally through casting.
 */
private[tensorframes] sealed abstract class ScalarTypeOperation[T : TypeTag : ClassTag] {
  /**
   * The SQL type associated with the given type.
   */
  val sqlType: NumericType

  /**
   * The TF type
   */
  val tfType: DataType

  /**
   * A zero element for this type
   */
  val zero: T

  /**
   * Returns a new converter for Array[Row] -> DenseTensor
   */
  def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[T]

  /**
   * The conversion of DenseTensor -> Row stuff.
   *
   * @param numElements the number of elements expected to be found in the tensor.
   */
  @deprecated("to remove", "now")
  def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any]

  /**
   * Defensive copy to an external array
   *
   * @param buff
   * @return
   */
  def convertBuffer(buff: ByteBuffer): mutable.WrappedArray[T]

  final def convertBuffer1(b0: Array[_], dim1: Int): Array[T] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1, (b.length, dim1))
    b
  }

  final def convertBuffer2(b0: Array[_], dim1: Int, dim2: Int): Array[Array[T]] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1 * dim2, (b.length, dim1, dim2))
    val res = Array.fill(dim1){ new Array[T](dim2) }
    var idx1 = 0
    while (idx1 < dim1) {
      var idx2 = 0
      while(idx2 < dim2) {
        // TODO(tjh) check math
        res(idx1)(idx2) = b(idx2 + dim2 * idx1)
        idx2 += 1
      }
      idx1 += 1
    }
    res
  }

  final def convertBuffer3(
      b0: Array[_], dim1: Int, dim2: Int, dim3: Int): Array[Array[Array[T]]] = {
    val b = b0.asInstanceOf[Array[T]]
    assert(b.length == dim1 * dim2 * dim3, (b.length, dim1, dim2, dim3))
    val res = Array.fill(dim1) { Array.fill(dim2) { new Array[T](dim3) } }
    var idx1 = 0
    while (idx1 < dim1) {
      var idx2 = 0
      while(idx2 < dim2) {
        var idx3 = 0
        while (idx3 < dim3) {
          // TODO(tjh) check math
          res(idx1)(idx2)(idx3) = b(idx3 + dim3 * idx2 + dim3 * dim2 * idx1)
          idx3 += 1
        }
        idx2 += 1
      }
      idx1 += 1
    }
    res
  }

  def tag: TypeTag[_] = implicitly[TypeTag[T]]
}

private[tensorframes] object SupportedOperations {
  private val ops: Seq[ScalarTypeOperation[_]] =
    Seq(DoubleOperations, IntOperations, LongOperations)

  val sqlTypes = ops.map(_.sqlType)

  private val tfTypes = ops.map(_.tfType)

  def opsFor(t: NumericType): ScalarTypeOperation[_] = {
    ops.find(_.sqlType == t).getOrElse {
      throw new IllegalArgumentException(s"Type $t is not supported. Only the following types are" +
        s"supported: ${sqlTypes.mkString(", ")}")
    }
  }

  def opsFor(t: DataType): ScalarTypeOperation[_] = {
    ops.find(_.tfType == t).getOrElse {
      throw new IllegalArgumentException(s"Type $t is not supported. Only the following types are" +
        s"supported: ${tfTypes.mkString(", ")}")
    }
  }

  def getOps[T : TypeTag](): ScalarTypeOperation[T] = {
    val ev: TypeTag[_] = implicitly[TypeTag[T]]
    ops.find(_.tag.tpe =:= ev.tpe).getOrElse {
      val tags = ops.map(_.tag.toString()).mkString(", ")
      throw new IllegalArgumentException(s"Type ${ev} is not supported. Only the following types " +
        s"are supported: ${tags}")
    }   .asInstanceOf[ScalarTypeOperation[T]]
  }

  def hasOps(x: Any): Boolean = x match {
    case _: Double => true
    case _: Int => true
    case _: Long => true
    case _ => false
  }
}

// ********** DOUBLES ************

private class DoubleTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Double](s, numCells) with Logging {
  private var _tensor: jtf.Tensor = null
  private var buffer: DoubleBuffer = null

  assert(! s.hasUnknown, s"Shape $s has unknown values.")

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logTrace(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_DOUBLE, physicalShape)
    logTrace(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
    buffer = byteBuffer().asDoubleBuffer()
    buffer.rewind()
  }

  override def appendRaw(d: Double): Unit = {
    buffer.put(d)
  }

  override def tensor(): jtf.Tensor = _tensor

  override def byteBuffer(): ByteBuffer =  _tensor.tensor_data().asByteBuffer()
}

private object DoubleOperations extends ScalarTypeOperation[Double] with Logging {
  override val sqlType = DoubleType
  override val tfType = DataType.DT_DOUBLE
  final override val zero = 0.0
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Double] =
    new DoubleTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asDoubleBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    logTrace(s"convertBuffer: dbuff: pos:${dbuff.position()}, cap:${dbuff.capacity()} " +
      s"limit:${dbuff.limit()} expected=$numElements")
    val res: Array[Double] = Array.fill(numBufferElements)(Double.NaN)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res

  }

  override def convertBuffer(buff: ByteBuffer): mutable.WrappedArray[Double] = {
    val dbuff = buff.asDoubleBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Double] = Array.fill(numBufferElements)(Double.NaN)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }
}

// ********** INT32 ************

private class IntTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Int](s, numCells) with Logging {
  private var _tensor: jtf.Tensor = null
  private var buffer: IntBuffer = null

  assert(! s.hasUnknown, s"Shape $s has unknown values.")

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logTrace(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_INT32, physicalShape)
    logTrace(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
    buffer =byteBuffer().asIntBuffer()
    buffer.rewind()
  }

  override def appendRaw(d: Int): Unit = {
    buffer.put(d)
  }

  override def tensor(): jtf.Tensor = _tensor

  override def byteBuffer(): ByteBuffer =  _tensor.tensor_data().asByteBuffer()
}

private object IntOperations extends ScalarTypeOperation[Int] with Logging {
  override val sqlType = IntegerType
  override val tfType = DataType.DT_INT32
  final override val zero = 0
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Int] =
    new IntTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asIntBuffer()
    dbuff.rewind()
    val res: Array[Int] = Array.fill(numElements)(Int.MinValue)
    dbuff.get(res)
    res
  }

  override def convertBuffer(buff: ByteBuffer): mutable.WrappedArray[Int] = {
    val dbuff = buff.asIntBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Int] = new Array[Int](numBufferElements)
    dbuff.get(res)
    res
  }
}

// ****** INT64 (LONG) ******

private class LongTensorConverter(s: Shape, numCells: Int)
  extends TensorConverter[Long](s, numCells) with Logging {
  private var _tensor: jtf.Tensor = null
  private var buffer: LongBuffer = null

  assert(! s.hasUnknown, s"Shape $s has unknown values.")

  override def reserve(): Unit = {
    logTrace(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logTrace(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_INT64, physicalShape)
    logTrace(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
    buffer = byteBuffer().asLongBuffer()
    buffer.rewind()
  }

  override def appendRaw(d: Long): Unit = {
    buffer.put(d)
  }

  override def tensor(): jtf.Tensor = _tensor

  override def byteBuffer(): ByteBuffer =  _tensor.tensor_data().asByteBuffer()
}

private object LongOperations extends ScalarTypeOperation[Long] with Logging {
  override val sqlType = LongType
  override val tfType = DataType.DT_INT64
  final override val zero = 0L
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Long] =
    new LongTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asLongBuffer()
    dbuff.rewind()
    val res: Array[Long] = Array.fill(numElements)(Long.MinValue)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }
  override def convertBuffer(buff: ByteBuffer): mutable.WrappedArray[Long] = {
    val dbuff = buff.asLongBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Long] = Array.fill(numBufferElements)(Long.MinValue)
    dbuff.get(res)
    logTrace(s"Extracted from buffer: ${res.toSeq}")
    res
  }
}