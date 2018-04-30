package com.datastax.demo.fraud

/**
  * Created by markwolters on 4/30/18.
  */
class RandomForestRegression {
  // RandomForest Regression
  import org.apache.spark.mllib.tree.RandomForest
  import org.apache.spark.mllib.tree.model.RandomForestModel
  import org.apache.spark.mllib.util.MLUtils

  val data = MLUtils.loadLibSVMFile(sc, "/data/cts.txt")
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))
  val numClasses = 2
  val categoricalFeaturesInfo = Map[Int, Int]()
  val numTrees = 3
  val featureSubsetStrategy = "auto"
  val impurity = "variance"
  val maxDepth = 4
  val maxBins = 32
  val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
  val labelsAndPredictions = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
  println("Test Mean Squared Error = " + testMSE)
  println("Learned regression forest model:\n" + model.toDebugString)
}
