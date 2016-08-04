package org.tensorframes

import org.apache.spark.sql.Row
import org.scalatest.FunSuite
import org.tensorframes.dsl.GraphScoping
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.{dsl => tf}
import org.tensorframes.dsl.Implicits._

class TrimmingOperationsSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging with GraphScoping {
  lazy val sql = sqlContext

  val ops = new DebugRowOps


  testGraph("Less rows in the output") {
    val df = make1(Seq(1.0, 2.0), "in")
    val out = tf.constant(Seq(1.0)) named "out"
    val df2 = df.mapBlocksTrimmed(out)
    assert(df2.schema.fieldNames === Seq("out"))
    assert(df2.collect() === Array(Row(1.0)))
  }

  testGraph("More rows in the output") {
    val df = make1(Seq(3.0), "in")
    val out = tf.constant(Seq(1.0, 2.0)) named "out"
    val df2 = df.mapBlocksTrimmed(out)
    assert(df2.schema.fieldNames === Seq("out"))
    assert(df2.collect() === Array(Row(1.0), Row(2.0)))
  }

  testGraph("As many rows in the output") {
    val df = make1(Seq(3.0, 4.0), "in")
    val out = tf.constant(Seq(1.0, 2.0)) named "out"
    val df2 = df.mapBlocksTrimmed(out)
    assert(df2.schema.fieldNames === Seq("out"))
    assert(df2.collect() === Array(Row(1.0), Row(2.0)))
  }

  testGraph("Less rows in the output in higher dimensions") {
    val df = make1(Seq(Seq(1.0), Seq(2.0)), "in").analyze()
    val out = tf.constant(Seq(Seq(1.0))) named "out"
    val df2 = df.mapBlocksTrimmed(out)
    assert(df2.schema.fieldNames === Seq("out"))
    assert(df2.collect() === Array(Row(Seq(1.0))))
  }

}
