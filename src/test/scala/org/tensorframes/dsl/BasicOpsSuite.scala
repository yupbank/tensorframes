package org.tensorframes.dsl

import org.apache.spark.{LoggingWrapper => Logging}
import org.scalatest.FunSuite
import org.tensorframes.dsl.ExtractNodes._
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

class BasicOpsSuite extends FunSuite with GraphScoping with Logging {


  testGraph("Add") {
    val x = tf.constant(1) named "x"
    val y = tf.constant(2) named "y"
    val z = x + y named "z"
    compareOutput(
      """
        |x = tf.constant(1, name='x')
        |y = tf.constant(2, name='y')
        |z = tf.add(x, y, name='z')
      """.stripMargin, z)
  }
}
