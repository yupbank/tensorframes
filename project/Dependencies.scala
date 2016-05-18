import sbt._
import sbtsparkpackage.SparkPackagePlugin
import sbtsparkpackage.SparkPackagePlugin.{autoImport => sp}
import Keys._

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.Locale

import xml.{NodeSeq, Node => XNode, Elem}
import xml.transform.{RuleTransformer, RewriteRule}

object Dependencies {
  // The spark version
  val targetSparkVersion = "1.6.1"

  val targetJCPPVersion = "1.2"

  val targetTensorFlowVersion = "0.8.0"

  def credentialPath: File = {
    Paths.get("sbtcredentials").toAbsolutePath().toFile
  }

  // If a custom version of tensorflow is available in lib, use this one. Otherwise use the
  // default version published in maven central.
  def customTF() = {
    val baseDir = new File(".")
    val f = baseDir / "lib" / s"javacpp-$targetJCPPVersion-tensorflow-$targetTensorFlowVersion-gpu"
    if (Files.exists(f.toPath)) {
      val f2 = f.getAbsoluteFile
      println(s"Using custom tensorflow version in $f2")
      unmanagedBase := f2
    } else {
      val vstring = s"$targetTensorFlowVersion-$targetJCPPVersion"
      // Add other versions here if necessary
      val packages = Seq(
        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring,
        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring classifier "linux-x86_64",
        "org.bytedeco.javacpp-presets" % "tensorflow" % vstring classifier "macosx-x86_64"
      )
      libraryDependencies ++= packages
    }
  }

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

  // These dependencies are explicitly bundled with the spark package, because they are pretty
  // hard to get right: protobuf3 needs to be shaded, and some environments like databricks do
  // not load dependencies with profiles.
  // The org.bytedeco could probably be removed.
  val blacklistedGroupIds = Set(
    "com.google.protobuf",
    "org.bytedeco",
    "org.bytedeco.javacpp-presets"
  )

  def dependenciesFilter(n: XNode) = new RuleTransformer(new RewriteRule {
    override def transform(n: XNode): NodeSeq = n match {
      case e: Elem if e.label == "dependency" &&
        blacklistedGroupIds.contains((e \ "groupId").text.trim) =>
        NodeSeq.Empty
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