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

  // ******* Control flow *********

  def scope[T](pathElem: String)(fun: => T): T = {
    Paths.withScope(pathElem)(fun)
  }

  /**
   * Used to express unknown dimensions.
   */
  val Unknown = Shape.Unknown

  // ******** Constants, Sequences, and Random Values ***********

  def placeholder[T : Numeric : TypeTag](shape: Int*): Operation = {
    val ops = SupportedOperations.getOps[T]()
    DslImpl.placeholder(ops.sqlType, Shape(shape: _*))
  }

  def constant[T : ConvertibleToDenseTensor](x: T): Operation = {
    val ev = implicitly[ConvertibleToDenseTensor[T]]
    build_constant(ev.tensor(x))
  }

  def zeros(shape: Int*): Operation = zeros[Float](shape:_*)

  def zeros[T : Numeric : ConvertibleToDenseTensor : TypeTag](shape: Int*): Operation = {
    fill(shape, implicitly[Numeric[T]].zero)
  }

  def ones(shape: Int*): Operation = ones[Float](shape: _*)

  def ones[T : Numeric : ConvertibleToDenseTensor : TypeTag](shape: Int*): Operation = {
    fill(shape, implicitly[Numeric[T]].one)
  }

  def fill[T : Numeric : ConvertibleToDenseTensor : TypeTag](
      dims: Seq[Int], value: T): Operation = {
    dims match {
      case Seq() => constant(value)
      case Seq(n) =>
        constant(Seq.fill(n)(value))
      case _ =>
        throw HighDimException(Shape(dims: _*))
    }
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


  // ******** Tensor Transformations ***********


  def identity(op: Operation): Operation = {
    build("Identity", parents = Seq(op))
  }

  def add(x: Operation, y: Operation): Operation =
    build("Add", parents=Seq(x, y), shapeInfer = broadcastShape)


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

