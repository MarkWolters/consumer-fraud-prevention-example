package com.datastax.demo.fraud

/**
  * Created by markwolters on 4/30/18.
  */
class RandomForestClassification {
  // RandomForest Classification
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
  val impurity = "gini"
  val maxDepth = 4
  val maxBins = 32

  val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
  val labelsAndPredictions = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testErr = labelsAndPredictions.filter(r => r._1 != r._2).count.toDouble / testData.count()
  println("Test Error = " + testErr)
  println("Learned regression forest model:\n" + model.toDebugString)
}
