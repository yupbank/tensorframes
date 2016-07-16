package org.tensorframes.dsl

import scala.collection.mutable

import org.tensorframes.Logging

/**
 * Operations that try to give a convenient way to express paths in expressions.
 *
 * This is very brittle and will *NOT* work in a multithreaded environment.
 */
// TODO(tjh) ensure that only one graph can be use at the same time.
private[dsl] object Paths extends Logging {
  private[this] var rpath: List[String] = Nil
  private[this] var counters: mutable.Map[String, Int] = mutable.Map.empty

  def withScope[T](s: String)(fun: => T): T = {
    rpath ::= s
    try {
      fun
    } finally {
      rpath = rpath.tail
    }
  }

  def withGraph[T](fun: => T): T = {
    val old = counters
    counters = mutable.Map.empty
    try {
      fun
    } finally {
      counters = old
    }
  }

  def creationPath(): List[String] = rpath

  private def path(l: List[String]): String = l.filterNot(_.isEmpty).reverse.mkString("/")

  def path(creationPath: List[String], requestedName: Option[String], opName: String): String = {
    val full = requestedName.getOrElse(opName).split("/").toList.reverse ::: creationPath
    val key = path(full)
    val c = {
      val before = counters.getOrElseUpdate(key, 0)
      counters.update(key, before + 1)
      before
    }

    logDebug(s"Request for $key -> $c")
    if (c == 0) {
      key
    } else {
      key + "_" + c
    }
  }
}
