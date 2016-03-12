package org.tensorframes

/**
 * In order to go around a limitation of the C++ kernel of TensorFlow, the shape of the outputs must be provided
 * to the interface
 */
// TODO: rename to TFExtraInfo (?)
// TODO: add the ordering of the column?
// It should also include: a specific subset of columns (if requested)
// Some data constants to be used to fill the placeholders
case class ShapeDescription(out: Map[String, Shape]) {
}

object ShapeDescription {
  val empty = ShapeDescription(Map.empty)
}