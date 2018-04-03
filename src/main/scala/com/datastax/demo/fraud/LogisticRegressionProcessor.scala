package com.datastax.demo.fraud

import java.io.{File, FileInputStream}
import java.util.Properties

import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.StreamingLogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.streaming.{Seconds, StreamingContext}


/**
  * Train a logistic regression model on the training stream of data and make predictions
  * on another stream, where the data streams arrive as text files into two different directories.
  *
  * As you add text files to `trainingDir` the model will continuously update.
  * Anytime you add text files to `testDir`, you'll see predictions from the current model.
  *
  * The rows of the text files must be labeled data points in the form `(y,[x1,x2,x3,...,xn])`
  * Where n is the number of features, y is a binary label, and n must be the same for train and test.
  *
  */
object LogisticRegressionProcessor {

  def main(args: Array[String]) {

    val prop = new Properties()
    val propsFile = new File(args(0).toString)
    if (propsFile.exists()) {
      prop.load(new FileInputStream(propsFile))
    } else {
      System.err.println("Properties file not found in resources")
      System.exit(1)
    }

    /**
      * First read in all the properties needed to run this example and create the
      * streaming context
      */
    val trainingDir = prop.getProperty("trainingDir")
    val testDir = prop.getProperty("testDir")
    val batchDuration = prop.getProperty("batchDuration").toLong
    val numFeatures = prop.getProperty("numFeatures").toInt

    val conf = new SparkConf().setMaster("local").setAppName("LogisticRegressionProcessor")
    val ssc = new StreamingContext(conf, Seconds(batchDuration))

    /**
      * Next map the training data and the test data to DStreams of LabeledPoint Objects
      */
    val trainingData = ssc.textFileStream(trainingDir).map(LabeledPoint.parse)
    val testData = ssc.textFileStream(testDir).map(LabeledPoint.parse)
    /**
      * Training data will have average price included already in order to train the processor
      * to find transactions where the cost is > 2 SD from the average.  RT transactions
      * (as represented by the test data set) will need to have this calculated by
      * determining the average purchase amount for this card.  If the amount of this purchase
      * falls outside of 2 SD flag it as potentially fraudulent.
      *
      * Checks could also be done based on location vs. where the card has been used in the past,
      * merchant category, etc. but we're keeping this example simple.
      */
    //val rawTestData = ssc.textFileStream(testDir).flatMap(_.split("\n"))
    //var account_no = ""
    //var testData = rawTestData.transform()

    //val transaction_history = ssc.cassandraTable("transactions", "credit_card_transactions").select("*")
    //  .where("account_no = ?", account_no)
    val model = new StreamingLogisticRegressionWithSGD()
      .setInitialWeights(Vectors.zeros(numFeatures))

    model.trainOn(trainingData)
    /**
      * Make a prediction and write results to DSE
      */
    model.predictOnValues(testData.map(lp => (lp.label, lp.features))).print()

    ssc.start()
    ssc.awaitTermination()

  }

}