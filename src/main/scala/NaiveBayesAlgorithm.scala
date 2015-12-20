package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class NBAlgorithmParams(
  lambda: Double
) extends Params

class NaiveBayesAlgorithm(val ap: NBAlgorithmParams)
  extends P2LAlgorithm[PreparedData, NaiveBayesModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): NaiveBayesModel = {
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    NaiveBayes.train(data.labeledPoints, ap.lambda)
  }

  def predict(model: NaiveBayesModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
