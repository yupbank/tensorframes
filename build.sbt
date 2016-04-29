// Your sbt build file. Guides on how to write one can be found at
// http://www.scala-sbt.org/0.13/docs/index.html

//import sbtprotobuf.{ProtobufPlugin=>PB}
import Dependencies._

name := "tensorframes"

//scalaVersion := "2.10.5"

crossScalaVersions := Seq("2.11.7", "2.10.5")

// Don't forget to set the version
version := "0.1.2"

// ******* Spark-packages settings **********

spName := "tjhunter/tensorframes"

sparkVersion := targetSparkVersion

// We need to manually build the artifact
//sparkComponents ++= Seq("core", "sql")

spIncludeMaven := false

licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

spShortDescription := "Tensorflow wrapper for DataFrames on Apache Spark"

spDescription := {
  """TensorFrames (TensorFlow on Spark Dataframes) lets you manipulate Spark's DataFrames with
    | TensorFlow programs.
    |
    |This package provides a small runtime to express and run TensorFlow computation graphs.
    |Tensorflow programs can be interpreted from:
    | - the official Python API
    | - the semi-official protocol buffer graph description format
    | - the Scala DSL embedded with TensorFrames (experimental)
    |
    |For more information, visit the TensorFrames user guide:
    |
  """.stripMargin
}

credentials += Credentials(credentialPath)

spIgnoreProvided := true

// *********** Regular settings ***********

// Using a custom resolver to host the TF artifacts, before publication
resolvers += 
  "Tensorframes-artifacts" at "https://github.com/tjhunter/tensorframes-artifacts/raw/master/deploy"


libraryDependencies += "org.apache.spark" %% "spark-core" % targetSparkVersion % "provided"

libraryDependencies += "org.apache.spark" %% "spark-sql" % targetSparkVersion % "provided"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.1.3" % "test"

libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.4"

// Compilation of proto files

//Seq(PB.protobufSettings: _*)

//protocOptions in PB.protobufConfig ++= Seq("--proto_path=/home/tensorframes/src/main/protobuf")

// Could not get protobuf to work -> manually adding it

libraryDependencies += "com.google.protobuf" % "protobuf-java" % "3.0.0-beta-1"

libraryDependencies += "org.tensorframes" % "javacpp" % jcppVersion

libraryDependencies += "org.tensorframes" % "tensorflow" % jcppVersion

libraryDependencies += "org.tensorframes" % "tensorflow-linux-x86_64" % jcppVersion

version in protobufConfig := "3.0.0-beta-1"

parallelExecution := false

javaOptions in run += "-Xmx6G"

assemblyShadeRules in assembly := Seq(
  ShadeRule.rename("com.google.protobuf.**" -> "org.tensorframes.protobuf3shade.@1").inAll
)

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

assemblyExcludedJars in assembly := {
  val cp = (fullClasspath in assembly).value
  val excludes = Set(
    "javacpp-1.2-SNAPSHOT-javadoc.jar",
    //"javacpp-tensorflow-linux-x86_64.jar", // This is too big to be included in the assembly
    "tensorflow-spark-tf-1.2-SNAPSHOT-javadoc.jar")
  cp filter { s => excludes.contains(s.data.getName) }
}

// Just add the python files in the final binary for now.

unmanagedResourceDirectories in Compile += {
  baseDirectory.value / "src/main/python/"
}

// Because of shading, sbt freaks out

addCommandAlias("doit", ";clean;compile;assembly")

// Remove all the dependencies from the pom because TF is packaged as a fat jar.
// See this stackoverflow question:
// http://stackoverflow.com/questions/24807875/how-to-remove-projectdependencies-from-pom
makePomConfiguration := makePomConfiguration.value.copy(process = dependenciesFilter)

// Spark packages messes this part
test in assembly := {}

Seq(tfPackageTask)
