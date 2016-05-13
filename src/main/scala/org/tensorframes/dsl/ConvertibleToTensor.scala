package org.tensorframes.dsl

import scala.reflect.runtime.universe.TypeTag

import org.tensorframes.impl.DenseTensor

/**
 * The basic trait for the type class that expresses conversions to a tensor.
 *
 * Implement this trait as an implicit class or object if you want to add custom support to
 * external data structure.
 */
trait ConvertibleToDenseTensor[T] {

  /**
   * Given an object of the given type, returns a dense tensor.
   */
  private[tensorframes] def tensor(data: T): DenseTensor
}

/**
 * Builtin conversions between standard types to dense tensors.
 */
trait DefaultConversions {

  implicit object DoubleConversion extends ConvertibleToDenseTensor[Double] {
    def tensor(data: Double): DenseTensor = DenseTensor(data)
  }

  implicit object FloatConversion extends ConvertibleToDenseTensor[Float] {
    // TODO(tjh) this is the wrong conversion here
    def tensor(data: Float): DenseTensor = DenseTensor(data.toDouble)
  }

  implicit object IntConversion extends ConvertibleToDenseTensor[Int] {
    def tensor(data: Int): DenseTensor = DenseTensor(data)
  }

  /**
   * Given a basic (numeric) type that is convertible to a dense tensor, this implicit transforms
   * expresses the fact that a sequence of these objects is also convertible to a dense tensor of
   * greater order.
   */
  implicit def sequenceVectorConversion[T : Numeric : TypeTag](
      implicit ev: ConvertibleToDenseTensor[T]): ConvertibleToDenseTensor[Seq[T]] = {
    new ConvertibleToDenseTensor[Seq[T]] {

      override private[tensorframes] def tensor(data: Seq[T]): DenseTensor = {
        DenseTensor(data)
      }
    }
  }

}

