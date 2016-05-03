package org.tensorframes.dsl

import org.tensorframes.impl.DenseTensor
import scala.reflect.runtime.universe.TypeTag


/**
 * The basic trait for the typeclass that expresses conversions to a tensor.
 */
trait ConvertibleToDenseTensor[T] {

  private[tensorframes] def tensor(data: T): DenseTensor
}

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

  implicit def sequenceVectorConversion[T : Numeric : TypeTag](
      implicit ev: ConvertibleToDenseTensor[T]): ConvertibleToDenseTensor[Seq[T]] = {
    new ConvertibleToDenseTensor[Seq[T]] {

      override private[tensorframes] def tensor(data: Seq[T]): DenseTensor = {
        DenseTensor(data)
      }
    }
  }

}

