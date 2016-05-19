package org.tensorframes.catalyst

import org.tensorframes.impl.MapBlocksSchema

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLContext, Strategy, TFHooks}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, UnaryNode}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.DataType

case class TestMapBlockPlan private[tensorframes](
    override val child: LogicalPlan,
    graphDefSerial: Broadcast[Array[Byte]],
    transform: MapBlocksSchema) extends UnaryNode {

  def output: Seq[Attribute] = {
    val dt: DataType = transform.outputSchema
    val attr = AttributeReference("obj", dt, nullable = false)()
    attr :: Nil
  }

  // TODO: this could be cleaned, based on the requested inputs
  override def references: AttributeSet = child.outputSet
}

case class TestMapBlockExec(
    child: SparkPlan,
    logicalPlan: TestMapBlockPlan) extends SparkPlan { // UnaryExecNode

  override def children: Seq[SparkPlan] = child :: Nil

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def doExecute(): RDD[InternalRow] = {
    child.execute().mapPartitions { it =>
      // This is where stuff happens
      ???
    }
  }

  override def output: Seq[Attribute] = logicalPlan.output

}

object TestStrategy extends Strategy {
  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case p: TestMapBlockPlan =>
      val sparkContext = SparkContext.getOrCreate()
      val session = SQLContext.getOrCreate(sparkContext).sparkSession
      val childPlan = TFHooks.planLater(session, p.child)
      TestMapBlockExec(childPlan, p) :: Nil
    case _ =>
      Nil
  }

  def ensureLoaded(): Unit = {
    assert(loaded_)
  }

  private lazy val loaded_ : Boolean = {
    val spark = SparkContext.getOrCreate()
    val sql = SQLContext.getOrCreate(spark)
    sql.experimental.extraStrategies ++= Seq(TestStrategy)
    true
  }
}

