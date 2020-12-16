name := "DB2DecisionTreeMLeap"

version := "0.1"

scalaVersion := "2.12.12"

libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.1"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.1" % "provided"
libraryDependencies += "ml.combust.mleap" %% "mleap-spark" % "0.16.0"

libraryDependencies += "com.ibm.db2" % "jcc" % "11.5.4.0"
