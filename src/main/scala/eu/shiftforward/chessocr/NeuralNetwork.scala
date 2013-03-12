package eu.shiftforward.chessocr

import org.encog.neural.networks.BasicNetwork
import org.encog.neural.networks.layers.BasicLayer
import org.encog.engine.network.activation.{ActivationSigmoid, ActivationLOG}
import org.encog.ml.data.MLDataSet
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation
import org.encog.ml.train.strategy.StopTrainingStrategy

object NeuralNetwork {
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
    // val score = new TrainingSetScore(trainingSet)
    // val trainAlt1 = new NeuralGeneticAlgorithm(network, new GaussianRandomizer(0.5, 0.5), score, 100, 0.1, 0.2)
    // val trainAlt2 = new NeuralSimulatedAnnealing(network, score, 10, 0.5, 100)
    // val trainMain = new Backpropagation(network, trainingSet, 0.000001, 0.0)
    val trainMain = new ResilientPropagation(network, trainingSet)

    val stop = new StopTrainingStrategy(0.000000001, 100)
    // trainMain.addStrategy(new HybridStrategy(trainAlt2, 0.01, 100, 200))
    trainMain.addStrategy(stop)

    var epoch = 0
    while (!stop.shouldStop()) {
      println("Training Epoch #" + epoch + " Error:" + trainMain.getError)
      trainMain.iteration(100)
      epoch += 100
    }

    trainMain.getError
  }
}
