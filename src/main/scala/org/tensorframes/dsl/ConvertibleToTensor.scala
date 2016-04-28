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
//    override def tag = implicitly[TypeTag[Double]]
  }

  implicit object IntConversion extends ConvertibleToDenseTensor[Int] {
    def tensor(data: Int): DenseTensor = DenseTensor(data)
//    override def tag = implicitly[TypeTag[Int]]
  }

  implicit def sequenceVectorConversion[T : Numeric : TypeTag](
      implicit ev: ConvertibleToDenseTensor[T]): ConvertibleToDenseTensor[Seq[T]] = {
    new ConvertibleToDenseTensor[Seq[T]] {

      override private[tensorframes] def tensor(data: Seq[T]): DenseTensor = {
//        implicit val t = ev.tag

        DenseTensor(data)
      }

//      override def tag = {
//        implicit val t = ev.tag
//        implicitly[TypeTag[Seq[T]]]
//      }
    }
  }

  implicit class RichConvertibleToDenseTensor[T](v: T)(implicit ev: ConvertibleToDenseTensor[T]) {
    def +(other: Operation): Operation = {
      constant(v) + other
    }
  }

}

