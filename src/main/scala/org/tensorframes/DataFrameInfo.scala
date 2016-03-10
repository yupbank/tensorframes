package org.tensorframes

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StructType}


class DataFrameInfo private (cs: Array[ColumnInformation]) extends Serializable {
  def cols: Seq[ColumnInformation] = cs

  def explain: String = {
    val els = cols.map { c =>
      c.stf.map { i =>
        s"${i.dataType.toString}${i.shape.toString}"
      } .getOrElse { "??" + DataFrameInfo.pprint(c.field.dataType) }
    }
    els.mkString("DataFrame[", ", ", "]")
  }

  /**
   * The information with the metadata merged in.
    *
    * @return
   */
  def merged: StructType = {
    StructType(cs.map(_.merged))
  }

  override def toString = explain
}

object DataFrameInfo {
  def pprint(s: DataType) = s.toString

  def apply(d: Seq[ColumnInformation]): DataFrameInfo = new DataFrameInfo(d.toArray)

  def get(df: DataFrame): DataFrameInfo = {
    new DataFrameInfo(df.schema.map(ColumnInformation.apply).toArray)
  }
}