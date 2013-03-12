import sbt._
import sbt.Keys._
import sbtassembly.Plugin._
import sbtassembly.Plugin.AssemblyKeys._

object ProjectBuild extends Build {
  lazy val project = "encog"

  lazy val root = Project(id = project,
                          base = file("."),
                          settings = Project.defaultSettings ++ assemblySettings)
                            .settings(
    organization := "eu.shiftforward",
    version := "0.1-SNAPSHOT",
    scalaVersion := "2.10.0",

    resolvers ++= Seq(
      "Typesafe Repository"           at "http://repo.typesafe.com/typesafe/releases/",
      "Typesafe Snapshots Repository" at "http://repo.typesafe.com/typesafe/snapshots/",
      "Sonatype Repository"           at "http://oss.sonatype.org/content/repositories/releases",
      "Sonatype Snapshots Repository" at "http://oss.sonatype.org/content/repositories/snapshots",
      "BerkeleyDB JE Repository"      at "http://download.oracle.com/maven/"
    ),

    libraryDependencies ++= Seq(
      "org.encog"                      % "encog-core"         % "3.1.+",
      "org.specs2"                    %% "specs2"             % "1.13"   % "test",
      "junit"                          % "junit"              % "4.11"   % "test"
    ),

    testOptions in Test += Tests.Argument(TestFrameworks.Specs2, "junitxml", "console"),

    scalacOptions ++= Seq("-deprecation", "-unchecked")
  )
}
