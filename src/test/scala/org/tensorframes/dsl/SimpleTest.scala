package org.tensorframes.dsl

import org.apache.spark.Logging
import org.scalatest.FunSuite
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

class SimpleTest extends FunSuite with Logging {

  import ExtractNodes._

  test("Simple test") {
    val x = tf.constant(3)
    compareOutput("tf.constant(3)", x)
  }
}
