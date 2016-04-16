package org.tensorframes.dsl

import org.apache.spark.sql.types.NumericType
import org.tensorflow.framework.{DataType, AttrValue}
import org.tensorframes.Shape

/**
 * Created by tjhunter on 3/22/16.
 */
private[dsl] object PrivateImplicits {

  private implicit class ShapeToAttr(s: Shape) {
    def toAttr: AttrValue = AttrValue.newBuilder().setShape(buildShape(s)).build()
  }

  private implicit class SQLTypeToAttr(s: NumericType) {
    def toAttr: AttrValue = buildType(s)
  }

  private implicit class DataTypeToAttr(dt: DataType) {
    def toAttr: AttrValue = dataTypeToAttrValue(dt)
  }

}
