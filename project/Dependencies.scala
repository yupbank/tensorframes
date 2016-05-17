import sbt._
import Keys._

import java.nio.file.Paths
import java.io.File
import java.nio.file.Files

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
}