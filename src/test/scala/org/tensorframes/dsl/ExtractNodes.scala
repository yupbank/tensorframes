package org.tensorframes.dsl

import java.io.{BufferedReader, InputStreamReader, File}
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets
import org.scalatest.ShouldMatchers

import scala.collection.JavaConverters._

object ExtractNodes extends ShouldMatchers {

  def executeCommand(py: String): Map[String, String] = {
    val content =
      s"""
         |import tensorflow as tf
         |$py
         |g = tf.get_default_graph().as_graph_def()
         |for n in g.node:
         |    print ">>>>>", str(n.name), "<<<<<<"
         |    print n
       """.stripMargin
    val f = File.createTempFile("pythonTest", "py")
    Files.write(f.toPath, content.getBytes(StandardCharsets.UTF_8))
    val p = new ProcessBuilder("myCommand", "myArg").start()
    val s = p.getInputStream
    val isr = new InputStreamReader(s)
    val br = new BufferedReader(isr)
    var res: String = ""
    var str: String = ""
    while(str != null) {
      str = br.readLine()
      if (str != null) {
        res = res + str
      }
    }
    println(res)
    res.split(">>>>>").map { b =>
      val zs = b.split("\n")
      val node = zs.head.drop(1).dropRight(5)
      val rest = zs.tail
      node -> rest.mkString("\n")
    } .toMap
  }

  def compareOutput(py: String, nodes: Operation*): Unit = {
    val g = TestUtilities.buildGraph(nodes.head, nodes.tail:_*)
    val m1 = g.getNodeList.asScala.map { n => n.getName -> n.toString } .toMap
    val pym = executeCommand(py)
    if ((m1.keySet -- pym.keySet).nonEmpty) {
      val diff = (m1.keySet -- pym.keySet).toSeq.sorted
      assert(false, s"Found extra nodes in scala: $diff")
    }
    if ((pym.keySet -- m1.keySet).nonEmpty) {
      val diff = (pym.keySet -- m1.keySet).toSeq.sorted
      assert(false, s"Found extra nodes in python: $diff")
    }
    for (k <- m1.keySet) {
      m1(k) should equal(pym(k))
    }
  }
}
