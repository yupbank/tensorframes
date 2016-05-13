package org.tensorframes.perf

import org.apache.spark.Logging
import org.apache.spark.sql.functions._
import org.scalatest.FunSuite
import org.tensorframes.TensorFramesTestSparkContext
import org.tensorframes.dsl.Implicits._
import org.tensorframes.dsl._

class PerformanceSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging {
  lazy val sql = sqlContext

  test("Perf 1") {
    val df = sql.range(0L, 20000000L).toDF("x")

    for (_ <- 1 to 10) {
      withGraph {
        val x = df.block("x", "x")
        val z = x + x named "z"
        val df2 = df.mapBlocks(z).select("x", "z")
        val res = df2.agg(sum("z")).collect()
        println(s"!!! res = $res")
      }
    }
  }
}
