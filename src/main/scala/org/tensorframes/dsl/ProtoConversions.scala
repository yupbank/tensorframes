package org.tensorframes.dsl

import org.apache.spark.sql.types.NumericType
import org.tensorflow.framework.{AttrValue, DataType, NodeDef}
import org.tensorframes.impl.SupportedOperations


/**
 * Utilities to convert data back and forth between the proto descriptions and the dataframe
 * descriptions.
 */
private[tensorframes] object ProtoConversions {
  def getDType(nodeDef: NodeDef): DataType = {
    val opt = Option(nodeDef.getAttr.get("T")).orElse(Option(nodeDef.getAttr.get("dtype")))
    val v = opt.getOrElse(throw new Exception(s"Neither 'T' no 'dtype' was found in $nodeDef"))
    v.getType
  }

  def getDType(sqlType: NumericType): DataType = {
    SupportedOperations.opsFor(sqlType).tfType
  }

  def sqlTypeToAttrValue(sqlType: NumericType): AttrValue = {
    AttrValue.newBuilder().setType(getDType(sqlType)).build()
  }

  def dataTypeToAttrValue(dataType: DataType): AttrValue = {
    AttrValue.newBuilder().setType(dataType).build()
  }

}
