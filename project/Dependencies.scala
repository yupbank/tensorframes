import sbt._
import java.nio.file.Paths
import java.io.File
import sbtsparkpackage.SparkPackagePlugin
import sbtsparkpackage.SparkPackagePlugin.{autoImport => sp}
import Keys._
import java.util.Locale
import scala.io.Source._

import xml.{NodeSeq, Node => XNode, Elem}
import xml.transform.{RuleTransformer, RewriteRule}

object Dependencies {
  // The spark version
  val targetSparkVersion = "1.6.1"

  def credentialPath: File = {
    Paths.get("sbtcredentials").toAbsolutePath().toFile
  }

//  lazy val rewrittenPomFile = TaskKey[File]("tf-rewrite-pom",
//    "Rewrites the POM file to remove deps")

//  lazy val rewrittenPomFileTask = rewrittenPomFile := {
//    val f: File = SparkPackagePlugin.autoImport.spMakePom.value
//    f
//  }

  val tfPackage = TaskKey[File]("tfPackage", "Packages TensorFrames")

  val tfPackageTask = tfPackage := {
    // This task could be broken down, but I do not have the working
    // knowledge of SBT to make this happen.
    val sparkVersion = sp.sparkVersion.value
    val sv = scalaVersion.value
    val v = version.value
    val ass = sbtassembly.AssemblyKeys.assembly.value
    val oldJar: File = sp.spPackage.value
    val oldDist: File = sp.spDist.value

    println("oldJar: " + oldJar.toString)
    println("oldDist: " + oldDist.toString)
    println("assembly: " + ass.toString)

    val spArtifactName = spBaseArtifactName(sp.spName.value,
      SparkPackagePlugin.packageVersion.value)
    val pom: File = sp.spMakePom.value

    val zipFile: File = sp.spDistDirectory.value / (spArtifactName + ".zip")

    IO.delete(zipFile)
    IO.zip(Seq(ass -> (spArtifactName + ".jar"), pom -> (spArtifactName + ".pom")), zipFile)
    println(s"\nZip File overwritten at: $zipFile\n")
    zipFile
  }

  def dependenciesFilter(n: XNode) = new RuleTransformer(new RewriteRule {
    override def transform(n: XNode): NodeSeq = n match {
      case e: Elem if e.label == "dependencies" => NodeSeq.Empty
      case other => other
    }
  }).transform(n).head

  private def spBaseArtifactName(sp: String, version: String): String = {
    val names = sp.split("/")
    require(names.length == 2,
      s"Please supply a valid Spark Package name. spName must be provided in " +
        s"the format: org_name/repo_name. Currently: $sp")
    require(names(0) != "abcdefghi" && names(1) != "zyxwvut",
      s"Please supply a Spark Package name. spName must be provided in " +
        s"the format: org_name/repo_name.")
    normalizeName(names(1)) + "-" + version
  }

  private def normalizeName(s: String) = s.toLowerCase(Locale.ENGLISH).replaceAll( """\W+""", "-")


}