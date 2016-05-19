package org.apache.spark

import org.apache.spark.internal.Logging

trait LoggingWrapper extends Logging {

  override protected def logInfo(msg: => String): Unit = {
    super.logInfo(msg)
  }

  override protected def logTrace(msg: => String): Unit = {
    super.logTrace(msg)
  }
}
