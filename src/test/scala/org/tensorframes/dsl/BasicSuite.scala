package org.tensorframes.dsl

import org.tensorframes.Logging
import org.scalatest.FunSuite
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

class BasicSuite extends FunSuite with GraphScoping with Logging {

  import ExtractNodes.compareOutput

  testGraph("Simple test") {
    val x = tf.constant(3)
    compareOutput("tf.constant(3)", x)
  }

  testGraph("Named constant") {
    val x = tf.constant(3) named "x"
    compareOutput("tf.constant(3, name='x')", x)
  }

  testGraph("Two constants with the same name") {
    val x = tf.constant(3)
    val x1 = tf.constant(3)
    compareOutput(
      """tf.constant(3)
        |tf.constant(3)
      """.stripMargin, x, x1)
  }

  testGraph("fill 1") {
    compareOutput("tf.fill([2],3)", tf.fill(Seq(2), 3))
  }
}
