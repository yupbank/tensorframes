package org.tensorframes

import java.nio.file.{Paths, Files}

import org.apache.spark.sql.DataFrame
import org.tensorflow.framework.GraphDef
import org.tensorframes.impl.{DenseTensor, SupportedOperations}

import scala.reflect.runtime.universe.TypeTag

/**
 * The public interface that reimplements a subset of the official TensorFlow python API.
 *
 * The name of the functions and of the parameters should be the same as in python.
 *
 * Note that elementary operations such as +, -, etc. can be directly expressed using the
 * standard mathematical operators.
 */
package object dsl {

  import DslImpl._

  private[dsl] implicit def op2Node(op: Operation): Node = op.asInstanceOf[Node]
  private[dsl] implicit def opSeq2NodeSeq(op: Seq[Operation]): Seq[Node] = {
    op.map(op2Node)
  }

  /**
   * Used to express unknown dimensions.
   */
  val Unknown = Shape.Unknown

  def placeholder[T : Numeric : TypeTag](shape: Int*): Operation = {
    val ops = SupportedOperations.getOps[T]()
    DslImpl.placeholder(ops.sqlType, Shape(shape: _*))
  }

  def constant[T : ConvertibleToDenseTensor](x: T): Operation = {
    val ev = implicitly[ConvertibleToDenseTensor[T]]
    build_constant(ev.tensor(x))
  }

  /**
   * Builds a block placeholder based on the content of a column in a dataframe.
   * @param df a dataframe
   * @param colName the name of a column in a dataframe
   * @return a placeholder
   */
  // TODO(tjh) make it work for column?
  def block(df: DataFrame, colName: String): Operation = {
    ???
  }

  def row(df: DataFrame, colName: String): Operation = {
    ???
  }

  def identity(op: Operation): Operation = {
    build("Identity", parents = Seq(op))
  }

  def add(x: Operation, y: Operation): Operation = build("Add", parents=Seq(x, y))


  // **** Reducers ******

  def reduce_min(
      input_tensor: Operation,
      reduction_indices: Seq[Int] = null): Operation = {
    build_reducer("Min", input_tensor, reduction_indices)
  }

  def reduce_sum(
      input_tensor: Operation,
      reduction_indices: Seq[Int] = null): Operation = {
    build_reducer("Sum", input_tensor, reduction_indices)
  }

}

