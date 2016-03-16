import sbt._
import java.nio.file.Paths

object Dependencies {
  // The spark version
  val targetSparkVersion = "1.6.0"

  // The current release of tensorflow being targeted.
  val tfVersion = "0.7.1"

  val jcppVersion = "0.0.2-0.7.1"

  def credentialPath: File = {
    Paths.get("sbtcredentials").toAbsolutePath().toFile
  }
}