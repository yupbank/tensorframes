package org.tensorframes

import com.typesafe.scalalogging.slf4j.{LazyLogging, StrictLogging}

private[tensorframes] trait Logging extends LazyLogging {
  def logDebug(s: String) = logger.debug(s)
  def logInfo(s: String) = logger.info(s)
  def logTrace(s: String) = logger.trace(s)
}
