package org.tensorframes.dsl

import org.apache.spark.Logging
import org.apache.spark.sql.types.NumericType
import org.tensorflow.framework.{NodeDef, AttrValue}
import org.tensorframes.{dsl => tf, ShapeDescription, Shape}
import scala.collection.JavaConverters._

/**
 * A node in the TensorFlow graph.
 */
trait Operation {
  /**
   * The path of the operation. It follows TensorFlow's path conventions.
 *
   * @return
   */
  def name: String

  /**
   * The name of the operation.
 *
   * @return
   */
  def opName: String

  /**
   * The dimensions of the default tensor output of this operation.
   *
   * @return
   */
  def dims: Seq[Int]

  def named(newName: String): Operation

  def +(other: Operation): Operation = tf.add(this, other)
}

/**
 * Implementation of an operation.
 */
private[tensorframes] case class Node(
    requestedName: Option[String], // The name given by the user
    creationPath: List[String], // The context path when this node was created
    opName: String,
    scalarType: NumericType,
    shape: Shape,
    parents: Seq[Node],
    internalParents: String => Seq[Node],
    isOp: Boolean,
    extraAttr: Map[String, AttrValue]) extends Operation with Logging {

  import ProtoConversions._
  import DslImpl._

  private var _path: String = null
  private var _createdParents: Seq[Node] = null

  logDebug(s"Created node $this")

  private def frozen = { _path != null }

  def freeze(): Unit = {
    logDebug(s"Calling freeze on $this with ${_path}")
    // May increment counter as a side effect here.
    if (frozen) {
      return
    }
    val p = Paths.path(creationPath, requestedName, opName)
    _path = p
    logDebug(s"freeze: Path $p for $this")
    val diff = creationPath.filterNot(_.isEmpty).mkString("/")
    val suffix = p.drop(diff.length)
    val ns = internalParents(suffix)
    ns.foreach(_.freeze())
    _createdParents = ns
    assert(frozen)
  }

  private def allParents = {
    require(frozen)
    parents ++ _createdParents
  }

  def name: String = {
    require(frozen)
    _path
  }

  override def dims: Seq[Int] = shape.dims.map(_.toInt)

  // The node and all its internal nodes
  def nodes: Seq[NodeDef] = {
    freeze()
    val b = NodeDef.newBuilder()
    b.setName(name)
    b.setOp(opName)
    allParents.foreach(p => b.addInput(p.name))
    if (isOp) {
      b.putAllAttr(Map("T" -> getDType(scalarType).toAttr).asJava)
    } else {
      b.putAllAttr(Map(
        "dtype" -> getDType(scalarType).toAttr).asJava)
    }
    b.putAllAttr(extraAttr.asJava)
    Seq(b.build()) ++ _createdParents.flatMap(_.nodes)
  }

  override def named(newName: String): Node = {
    val c = copy(requestedName = Some(newName))
    c.freeze()
    c
  }

}

private[dsl] object Node {

  def apply(requestedName: Option[String],
            opName: String,
            scalarType: NumericType,
            shape: Shape,
            parents: Seq[Node],
            internalParents: String => Seq[Node],
            isOp: Boolean,
            extraAttr: Map[String, AttrValue]) = {
    val p = Paths.creationPath()
    new Node(requestedName, p, opName, scalarType, shape, parents, internalParents, isOp, extraAttr)
  }

  def hints(ns: Seq[Node]): ShapeDescription = {
    val m = ns.map { n =>
      n.name -> n.shape
    } .toMap
    val f = ns.map(_.name)
    ShapeDescription(m, f)
  }
}
