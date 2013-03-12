package eu.shiftforward.chessocr

import java.io.File
import org.encog.ml.data.basic.{BasicMLData, BasicMLDataSet}
import org.encog.persist.EncogDirectoryPersistence

object ChessOCR extends App {
  import NeuralNetwork._
  import Pieces._
  import ImageManipulation._

  implicit val imageSize = 32
  implicit val pathPrefix = "data//"

  val trainData = TrainData.trainData(imageSize, pathPrefix)

  val testData = List(
    (processImage("1361115685136_57.jpg"), whiteKnight)
  )

  val trainingSet = new BasicMLDataSet(
    trainData.map(_._1.map(v => 255.0 / v)).toArray,
    trainData.map(_._2).toArray
  )

  val network = opticalNetwork(imageSize * imageSize, 10, 7)
  val error = trainNetwork(network, trainingSet)

  EncogDirectoryPersistence.saveObject(new File("neural-trained.eg"), network)

  testData.foreach { test =>
    println(network.compute(new BasicMLData(test._1.map(v => 255.0 / v))).getData.toVector)
  }
}
