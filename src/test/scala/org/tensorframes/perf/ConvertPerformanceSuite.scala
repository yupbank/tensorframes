package org.tensorframes.perf

import org.bytedeco.javacpp.{tensorflow => jtf}
import org.scalatest.FunSuite

import org.apache.spark.Logging
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.tensorframes.{ColumnInformation, Shape, TensorFramesTestSparkContext}
import org.tensorframes.impl.{DataOps, SupportedOperations}

class ConvertPerformanceSuite
  extends FunSuite with TensorFramesTestSparkContext with Logging {

  lazy val sql = sqlContext

  def println(s: String): Unit = java.lang.System.err.println(s)

  def f[T](a: T, b: T): T = a

  def getTensor(
      sqlType: NumericType,
      row: Row,
      cellShape: Shape,
      numCells: Int): jtf.TensorVector = {
    val conv = SupportedOperations.opsFor(sqlType).tfConverter(cellShape, numCells)
    conv.reserve()
    (0 until numCells).foreach { _ => conv.append(row, 0) }
    val t = conv.tensor()
    val tv = new jtf.TensorVector(1)
    tv.put(0L, t)
    tv
  }

  ignore("performance of convert - int1") {
    val numCells = 10000000
    val rows = (0 until numCells).map { i =>
      Row(1)
    } .toArray
    val schema = StructType(Seq(StructField("f1", IntegerType, nullable = false)))
    println("generated data")
    logInfo("generated data")
    val start = System.nanoTime()
    val numIters = 100
    var x: Int = 0
    for (_ <- 1 to numIters) {
      x += DataOps.convert(rows, schema, Array(0), fastPath = true).sizeof()
    }
    val end = System.nanoTime()
    val tIter = (end - start) / (1e9 * numIters)
    println(s"perf: $tIter s / call")
    logInfo(s"perf: $tIter s / call")
  }

  ignore("performance of convert - int[1000]") {
    val numVals = 10000000
    val numCells = 1
    // Creating the rows this way, because we need to respect the collection used by Spark when
    // unpacking the rows.
    val rows = sqlContext.createDataFrame(Seq.fill(numCells)(Tuple1(Seq.fill(numVals)(1)))).collect()
    val schema = StructType(Seq(ColumnInformation.structField("f1", IntegerType,
      Shape(numCells, numVals))))
    println("generated data")
    logInfo("generated data")
    val start = System.nanoTime()
    val numIters = 100
    var x: Int = 0
    for (_ <- 1 to numIters) {
      x += DataOps.convert(rows, schema, Array(0), fastPath = true).sizeof()
    }
    val end = System.nanoTime()
    val tIter = (end - start) / (1e9 * numIters)
    println(s"perf: $tIter s / call")
    logInfo(s"perf: $tIter s / call")
  }

}
