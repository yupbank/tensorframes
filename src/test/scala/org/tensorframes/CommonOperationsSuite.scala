package org.tensorframes

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.scalatest.FunSuite
import org.tensorframes.impl.{SupportedOperations, ScalarTypeOperation, DebugRowOps}

import scala.reflect.runtime.universe._


// This is a fairly complicated base class that include a lot of machinery to express tests
// in doubles, and automatically replicate these tests for other numerical types.
abstract class CommonOperationsSuite[T](implicit val ev1: Numeric[T],
                                        implicit val ev2: TypeTag[T])
  extends FunSuite with TensorFramesTestSparkContext with Logging {
  import Shape.Unknown

  val Unknown = Shape.Unknown
  val ops = new DebugRowOps
  lazy val sql = sqlContext
  lazy val dataOps: ScalarTypeOperation[_] = SupportedOperations.getOps[T]
  lazy val dtype = dataOps.sqlType
  lazy val dtname = dtype.typeName

  def convert(x: Double): T

  trait Converter[A, B] {
    def conv(a: A): B
  }

  private object Converter {
    def apply[A, B](f: A => B): Converter[A, B] = new Converter[A, B] {
      override def conv(a: A): B = f(a)
    }
  }

  implicit class RichConversion[A](a: A) {
    def u[B](implicit c: Converter[A, B]): B = c.conv(a)
  }

  implicit object C1 extends Converter[Double, T] {
    def conv(a: Double): T = convert(a)
  }

  implicit object R extends Converter[Row, Row] {
    def conv(r: Row): Row = {
      val vs = (0 until r.length).map { i => c(r.get(i)) }
      Row(vs: _*)
    }
  }

  implicit def makeConverterTuple1[A1, B1]
  (implicit ev1: Converter[A1, B1]): Converter[Tuple1[A1], Tuple1[B1]] = {
    Converter { z: Tuple1[A1] => Tuple1(ev1.conv(z._1)) }
  }

  implicit def makeConverterStringTuple2[A1, B1]
  (implicit ev1: Converter[A1, B1]): Converter[Tuple2[String, A1], Tuple2[String, B1]] = {
    Converter { z: Tuple2[String, A1] => Tuple2(z._1, ev1.conv(z._2)) }
  }

  implicit def makeConverterTuple2[A1, A2, B1, B2]
  (implicit ev1: Converter[A1, B1], ev2: Converter[A2, B2]): Converter[(A1, A2), (B1, B2)] = {
    Converter { z: Tuple2[A1, A2] => Tuple2(ev1.conv(z._1), ev2.conv(z._2)) }
  }

  implicit def makeConverterSeq[A, B](implicit ev: Converter[A, B]): Converter[Seq[A], Seq[B]] = {
    Converter { xs: Seq[A] => xs.map(ev.conv) }
  }

  private def c(x: Any): Any = x match {
    case d: Double => d
    case s: Seq[_] => s.map(c)
    case (a1, b1) => c(a1) -> c(b1)
    case a: Any => a
  }

  implicit class C3[A](a: Seq[A]) {
    def u[B](implicit ev: Converter[A, B]): Seq[B] = a.map(ev.conv)
  }

  implicit class C2[A1, A2](z:(A1, A2)) {
    def u[B1, B2](implicit ev1: Converter[A1, B1], ev2: Converter[A2, B2]): (B1, B2) = {
      ev1.conv(z._1) -> ev2.conv(z._2)
    }
  }

}