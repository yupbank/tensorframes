resolvers += "Spark Packages repo" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.4")

// You need protoc3 for this to work
addSbtPlugin("com.github.gseitz" % "sbt-protobuf" % "0.5.3")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.3")