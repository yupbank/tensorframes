package org.tensorframes

import org.scalatest.FunSuite
import org.tensorframes.impl.DenseTensor

class DenseTensorSuite extends FunSuite {

  test("Serialization of doubles and endianness") {
    val d = DenseTensor(1.0)
    val p = DenseTensor.toTensorProto(d)
    assert(p.getDoubleVal(0) === 1.0)
  }

  test("Serialization of ints and endianness") {
    val d = DenseTensor(1)
    val p = DenseTensor.toTensorProto(d)
    assert(p.getIntVal(0) === 1)
  }
}
