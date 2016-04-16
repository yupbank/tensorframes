package org.tensorframes.dsl

import org.apache.spark.Logging
import org.apache.spark.sql.types.NumericType
import org.tensorflow.framework.{TensorShapeProto, DataType, AttrValue, GraphDef}
import org.tensorframes.Shape
import org.tensorframes.impl.DenseTensor
import org.tensorframes.test.ProtoConversions

object Conversions {
}

private[dsl] object DslImpl extends Logging {
  import ProtoConversions._

  private var counter: Int = 0

  private[dsl] implicit class ShapeToAttr(s: Shape) {
    def toAttr: AttrValue = AttrValue.newBuilder().setShape(buildShape(s)).build()
  }

  private[dsl] implicit class SQLTypeToAttr(s: NumericType) {
    def toAttr: AttrValue = buildType(s)
  }

  private[dsl] implicit class DataTypeToAttr(dt: DataType) {
    def toAttr: AttrValue = dataTypeToAttrValue(dt)
  }

  def buildShape(s: Shape): TensorShapeProto = s.toProto

  private def buildType(sqlType: NumericType): AttrValue = {
    AttrValue.newBuilder().setType(getDType(sqlType)).build()
  }


  def buildGraph(nodes: Seq[Node]): GraphDef = {
    logDebug(s"buildGraph for nodes: ${nodes.map(_.name)}")
    var treated: Map[String, Node] = Map.empty
    nodes.foreach { n =>
      logDebug(s"call: n=${n.name}, treated=${treated.keySet}")
      treated = getClosure(n, treated)
    }
    val b = GraphDef.newBuilder()
    treated.values.map(_.node).foreach(b.addNode)
    b.build()
  }

  private def getClosure(node: Node, treated: Map[String, Node]): Map[String, Node] = {
    logDebug(s"closure: n=${node.name}, parents=${node.parents.map(_.name)}," +
      s" treated=${treated.keySet}")
    val explored = node.parents
      .filterNot(n => treated.contains(n.node.getName))
      .flatMap(getClosure(_, treated + (node.name -> node)))
      .toMap

    uniqueByName(node +: (explored.values.toSeq ++ treated.values.toSeq))
  }

  private def uniqueByName(nodes: Seq[Node]): Map[String, Node] = {
    nodes.groupBy(_.name).mapValues(_.head)
  }

  def build_constant(dt: DenseTensor): Node = {
    val a = AttrValue.newBuilder().setTensor(DenseTensor.toTensorProto(dt))
    build("Const", isOp = false,
      shape = dt.shape, dtype = dt.dtype,
      extraAttrs = Map("value" -> a.build()))
  }

  private[tensorframes] def placeholder(dtype: NumericType, shape: Shape): Node = {
    build("Placeholder", shape=shape, dtype=dtype, isOp = false,
      extraAttrs = Map("shape" -> shape.toAttr))
  }

  private def commonShape(shapes: Seq[Shape]): Shape = {
    require(shapes.nonEmpty)
    require(shapes.forall(_ == shapes.head), s"$shapes")
    shapes.head
  }

  private def commonType(dtypes: Seq[NumericType]): NumericType = {
    require(dtypes.nonEmpty)
    require(dtypes.forall(_ == dtypes.head))
    dtypes.head
  }

  def build(
      opName: String,
      name: String = null,
      parents: Seq[Node] = Seq.empty,
      isOp: Boolean = true,
      dtype: NumericType = null,
      shape: Shape = null,
      dtypeInfer: Seq[NumericType] => NumericType = commonType,
      shapeInfer: Seq[Shape] => Shape = commonShape,
      extraAttrs: Map[String, AttrValue] = Map.empty): Node = {
    val n = Option(name).getOrElse {
      counter += 1
      s"${opName}_$counter"
    }
    val dt = Option(dtype).getOrElse(dtypeInfer(parents.map(_.scalarType)))
    val sh = Option(shape).getOrElse(shapeInfer(parents.map(_.shape)))
    Node(n, opName, dt, sh, parents, isOp, extraAttrs)
  }





  // Reducers (unfinished business)

  def reduce_min(
                  input_tensor: Node,
                  reduction_indices: Seq[Int] = null,
                  name: String = null): Node = {
    build_reducer("Min", input_tensor, reduction_indices, name)
  }

  def reduce_sum(
                  input_tensor: Node,
                  reduction_indices: Seq[Int] = null,
                  name: String = null): Node = {
    build_reducer("Sum", input_tensor, reduction_indices, name)
  }

  def build_reducer(
                             opName: String,
                             parent: Node,
                             reduction_indices: Seq[Int] = null,
                             name: String = null): Node = {
    val idxs = constant(reduction_indices) named (parent.name + "/reduction_indices")
    val attr = AttrValue.newBuilder().setB(false).build()
    build(opName, name, Seq(parent, idxs),
      dtype = parent.scalarType,
      shape = reduce_shape(parent.shape, Option(reduction_indices).getOrElse(Nil)),
      extraAttrs = Map("keep_dims" -> attr))
  }

  private def reduce_shape(s: Shape, red_indices: Seq[Int]): Shape = {
    require(s.numDims >= red_indices.size)
    // Special case for empty
    if (red_indices.isEmpty) {
      Shape.empty
    } else {
      // The remaining dimensions:
      val rem = s.dims.indices.filterNot(red_indices.contains)
      Shape(rem: _*)
    }
  }

}