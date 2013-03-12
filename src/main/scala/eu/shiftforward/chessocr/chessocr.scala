package chess

import java.awt.image.{RescaleOp, DataBufferByte, BufferedImage}
import javax.imageio.ImageIO
import java.io.File
import org.encog.neural.networks.BasicNetwork
import org.encog.ml.data.MLDataSet
import org.encog.neural.networks.training.TrainingSetScore
import org.encog.neural.networks.training.genetic.NeuralGeneticAlgorithm
import org.encog.mathutil.randomize.GaussianRandomizer
import org.encog.neural.networks.training.propagation.back.Backpropagation
import org.encog.ml.train.strategy.{HybridStrategy, StopTrainingStrategy}
import org.encog.neural.networks.layers.BasicLayer
import org.encog.engine.network.activation.{ActivationLOG, ActivationSigmoid}
import org.encog.ml.data.basic.{BasicMLData, BasicMLDataSet}
import org.encog.neural.networks.training.anneal.NeuralSimulatedAnnealing
import java.awt.RenderingHints
import scala.collection.JavaConversions._
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation
import org.encog.persist.EncogDirectoryPersistence
import org.encog.neural.thermal.BoltzmannMachine

object ChessOCR extends App {
  val imageSize = 32

  def opticalNetwork(input: Int, hidden: Int, output: Int) = {
    val network = new BasicNetwork
    network.addLayer(new BasicLayer(null, true, input))
  	network.addLayer(new BasicLayer(new ActivationLOG, true, hidden))
    network.addLayer(new BasicLayer(new ActivationSigmoid, false, output))
  	network.getStructure.finalizeStructure()
  	network.reset()
    network
  }

  def trainNetwork(network: BasicNetwork, trainingSet: MLDataSet) = {
    val score = new TrainingSetScore(trainingSet)
  	val trainAlt1 = new NeuralGeneticAlgorithm(network, new GaussianRandomizer(0.5, 0.5), score, 100, 0.1, 0.2)
    val trainAlt2 = new NeuralSimulatedAnnealing(network, score, 10, 0.5, 100)
  	// val trainMain = new Backpropagation(network, trainingSet, 0.000001, 0.0)
    val trainMain = new ResilientPropagation(network, trainingSet)

    val stop = new StopTrainingStrategy(0.000000001, 100)
  	trainMain.addStrategy(new HybridStrategy(trainAlt2, 0.01, 100, 200))
    trainMain.addStrategy(stop)

  	var epoch = 0
  	while (!stop.shouldStop()) {
  		trainMain.iteration()
      if (epoch % 100 == 0) println("Training Epoch #" + epoch + " Error:" + trainMain.getError)
  		epoch += 1
  	}

  	trainMain.getError
  }

  def processImage(fileName: String)(implicit pathPrefix: String = ""): Array[Byte] = {
    val originalImage = ImageIO.read(new File(pathPrefix + fileName))
    val resizedImage = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_BYTE_GRAY)
    val g = resizedImage.createGraphics()
    g.setRenderingHints(Map(RenderingHints.KEY_RENDERING -> RenderingHints.VALUE_RENDER_QUALITY))
    g.setRenderingHints(Map(RenderingHints.KEY_INTERPOLATION -> RenderingHints.VALUE_INTERPOLATION_BICUBIC))
    g.setRenderingHints(Map(RenderingHints.KEY_ANTIALIASING -> RenderingHints.VALUE_ANTIALIAS_ON))
    g.drawImage(originalImage, 0, 0, imageSize, imageSize, null)
    g.dispose()

    val rescaleOp = new RescaleOp(2f, 1, null)
    rescaleOp.filter(resizedImage, resizedImage)

    ImageIO.write(resizedImage, "jpg", new File(pathPrefix + "new" + fileName))
    resizedImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
  }

  implicit val pathPrefix = "data//"

  // output is pawn, tower, bishop, knight, queen, king, color (white)
  val trainData = List(
    (processImage("1361115508850_0.png"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361115508850_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361115508850_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115508850_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361115508850_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361115508850_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115508850_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
    (processImage("1361115508850_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),

    (processImage("1361115685136_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361115685136_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361115685136_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115685136_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361115685136_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361115685136_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115685136_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361115685136_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),

    (processImage("1361115952056_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361115952056_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361115952056_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115952056_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361115952056_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361115952056_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115952056_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361115952056_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),

    (processImage("1361115817053_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361115817053_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361115817053_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115817053_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361115817053_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361115817053_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115817053_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)),
    (processImage("1361115817053_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),

    (processImage("1361115873402_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361115873402_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361115873402_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361115873402_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361115873402_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361115873402_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361115873402_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361115873402_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),

    (processImage("1361116101610_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361116101610_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361116101610_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116101610_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361116101610_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361116101610_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116101610_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361116101610_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),

    (processImage("1361116176611_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361116176611_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361116176611_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116176611_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361116176611_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361116176611_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116176611_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361116176611_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0)),

    (processImage("1361116557698_0.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_1.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_2.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_3.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    (processImage("1361116557698_4.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
    (processImage("1361116557698_5.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_6.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_7.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_8.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_9.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    (processImage("1361116557698_54.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_55.jpg"), Array(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_56.jpg"), Array(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_58.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_59.jpg"), Array(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    (processImage("1361116557698_60.jpg"), Array(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)),
    (processImage("1361116557698_61.jpg"), Array(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)),
    (processImage("1361116557698_62.jpg"), Array(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
    (processImage("1361116557698_63.jpg"), Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
  )

  val testData = List(
    (processImage("1361115685136_57.jpg"), Array(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0))
  )

  val trainingSet = new BasicMLDataSet(
    trainData.map(_._1.map(v => 255.0 / v)).toArray,
    trainData.map(_._2).toArray
  )

  val network = opticalNetwork(imageSize*imageSize, 10, 7)
  val error = trainNetwork(network, trainingSet)

  EncogDirectoryPersistence.saveObject(new File("neural-trained.eg"), network)

  testData.foreach { test =>
    println(network.compute(new BasicMLData(test._1.map(v => 255.0 / v))).getData.toVector)
  }
}
