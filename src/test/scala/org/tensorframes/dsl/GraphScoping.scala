package org.tensorframes.dsl

import org.scalatest.FunSuite

import org.tensorframes.{dsl => tf}


trait GraphScoping { self: FunSuite =>
  import tf.withGraph

  def testGraph(s: String)(fun: => Unit): Unit = {
    test(s) { withGraph { fun } }
  }

}
