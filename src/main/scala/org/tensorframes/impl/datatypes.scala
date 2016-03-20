package org.tensorframes.impl

import java.nio._

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType, NumericType}
import org.bytedeco.javacpp.{tensorflow => jtf}
import org.tensorflow.framework.DataType
import org.tensorframes.Shape

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
        row.getSeq[T](position).foreach(appendRaw)
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
//    logDebug(s"toByteArray: lim=${buff.limit()} size=${buff.position()}")
    val res = Array.fill[Byte](buff.limit())(0)
    buff.get(res, 0, buff.limit())
//    (0 until buff.limit()).foreach(i => buff.get)
//    logDebug(s"toByteArray: res=${res.toSeq}")
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
private[tensorframes] sealed abstract class ScalarTypeOperation[T : TypeTag] {
  /**
   * The SQL type associated with the given type.
   */
  val sqlType: NumericType

  /**
   * The TF type
   */
  val tfType: DataType

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

  def convertBuffer(buff: ByteBuffer): IndexedSeq[Any]

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
    logDebug(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logDebug(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_DOUBLE, physicalShape)
    logDebug(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
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
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Double] =
    new DoubleTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asDoubleBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    logDebug(s"convertBuffer: dbuff: pos:${dbuff.position()}, cap:${dbuff.capacity()} " +
      s"limit:${dbuff.limit()} expected=$numElements")
//    val res = new mutable.ArrayBuffer[Double]()
    val res: Array[Double] = Array.fill(numBufferElements)(Double.NaN)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
    res

  }

  override def convertBuffer(buff: ByteBuffer): IndexedSeq[Any] = {
    val dbuff = buff.asDoubleBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Double] = Array.fill(numBufferElements)(Double.NaN)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
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
    logDebug(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logDebug(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_INT32, physicalShape)
    logDebug(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
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
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Int] =
    new IntTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asIntBuffer()
    dbuff.rewind()
    val res: Array[Int] = Array.fill(numElements)(Int.MinValue)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
    res
  }
  override def convertBuffer(buff: ByteBuffer): IndexedSeq[Any] = {
    val dbuff = buff.asIntBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Int] = Array.fill(numBufferElements)(Int.MinValue)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
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
    logDebug(s"Reserving for $numCells units of shape $shape")
    val s2 = s.prepend(numCells)
    val physicalShape = TensorFlowOps.shape(s2)
    logDebug(s"s2=$s2 phys=${TensorFlowOps.jtfShape(physicalShape)}")
    _tensor = new jtf.Tensor(jtf.TF_INT64, physicalShape)
    logDebug(s"alloc=${TensorFlowOps.jtfShape(_tensor.shape())}")
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
  override val sqlType = IntegerType
  override val tfType = DataType.DT_INT64
  override def tfConverter(cellShape: Shape, numCells: Int): TensorConverter[Long] =
    new LongTensorConverter(cellShape, numCells)
  override def convertBuffer(buff: ByteBuffer, numElements: Int): Iterable[Any] = {
    val dbuff = buff.asLongBuffer()
    dbuff.rewind()
    val res: Array[Long] = Array.fill(numElements)(Long.MinValue)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
    res
  }
  override def convertBuffer(buff: ByteBuffer): IndexedSeq[Any] = {
    val dbuff = buff.asLongBuffer()
    dbuff.rewind()
    val numBufferElements = dbuff.limit() - dbuff.position()
    val res: Array[Long] = Array.fill(numBufferElements)(Long.MinValue)
    dbuff.get(res)
    logDebug(s"Extracted from buffer: ${res.toSeq}")
    res
  }
}