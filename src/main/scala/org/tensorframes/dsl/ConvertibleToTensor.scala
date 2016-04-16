package org.tensorframes.dsl

import org.tensorframes.impl.DenseTensor

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

  implicit object IntConversion extends ConvertibleToDenseTensor[Int] {
    def tensor(data: Int): DenseTensor = DenseTensor(data)
  }

}

