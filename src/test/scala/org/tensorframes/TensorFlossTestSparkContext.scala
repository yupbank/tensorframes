package org.tensorframes

import scala.reflect.runtime.universe._

import org.scalatest.{FunSuite, BeforeAndAfterAll}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, DataFrame, SQLContext}

trait TensorFramesTestSparkContext extends BeforeAndAfterAll { self: FunSuite =>
  @transient var sc: SparkContext = _
  @transient var sqlContext: SQLContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[1]")
      .setAppName("TensorFramesTest")
      .set("spark.sql.shuffle.partitions", "4")  // makes small tests much faster
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
  }

  override def afterAll() {
    sqlContext = null
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  def make1[T : TypeTag](xs: Seq[T], col: String): DataFrame = {
    sqlContext.createDataFrame(xs.map(Tuple1.apply)).toDF(col)
  }

  def compareRows(r1: Array[Row], r2: Seq[Row]): Unit = {
    val a = r1.sortBy(_.toString())
    val b = r2.sortBy(_.toString())
    assert(a === b)
  }


}