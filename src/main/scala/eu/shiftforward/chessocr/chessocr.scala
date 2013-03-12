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
    (processImage("1361116557698_0.jpg"), blackTower),
    (processImage("1361116557698_1.jpg"), blackKnight),
    (processImage("1361116557698_2.jpg"), blackBishop),
    (processImage("1361116557698_3.jpg"), blackQueen),
    (processImage("1361116557698_4.jpg"), blackKing),
    (processImage("1361116557698_5.jpg"), blackBishop),
    (processImage("1361116557698_6.jpg"), blackKnight),
    (processImage("1361116557698_7.jpg"), blackTower),
    (processImage("1361116557698_8.jpg"), blackPawn),
    (processImage("1361116557698_9.jpg"), blackPawn),
    (processImage("1361116557698_54.jpg"), whitePawn),
    (processImage("1361116557698_55.jpg"), whitePawn),
    (processImage("1361116557698_56.jpg"), whiteTower),
    (processImage("1361116557698_57.jpg"), whiteKnight),
    (processImage("1361116557698_58.jpg"), whiteBishop),
    (processImage("1361116557698_59.jpg"), whiteQueen),
    (processImage("1361116557698_60.jpg"), whiteKing),
    (processImage("1361116557698_61.jpg"), whiteBishop),
    (processImage("1361116557698_62.jpg"), whiteKnight),
    (processImage("1361116557698_63.jpg"), whiteTower)
  )

  val trainingSet = new BasicMLDataSet(
    trainData.map(_._1.map(v => 255.0 / v)).toArray,
    trainData.map(_._2).toArray
  )

  val network = opticalNetwork(imageSize * imageSize, 7, 7)
  val error = trainNetwork(network, trainingSet)

  EncogDirectoryPersistence.saveObject(new File("neural-trained.eg"), network)

  testData.foreach { test =>
    println(network.compute(new BasicMLData(test._1.map(v => 255.0 / v))).getData.toVector.map(x => if (x >= 0.5) 1 else 0))
  }
}
