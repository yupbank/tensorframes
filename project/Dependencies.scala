import sbtsparkpackage.SparkPackagePlugin.{autoImport => sp}

import scala.xml.{Node => XNode}

object Dependencies {
  // The spark version
  val targetSparkVersion = sys.props.getOrElse("spark.version", "2.3.0")
  val targetTensorFlowVersion = "1.6.0"

}