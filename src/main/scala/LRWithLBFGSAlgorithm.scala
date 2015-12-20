package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class AlgorithmParams(
  numClasses: Int,
  intercept: Boolean 
) extends Params

class LRWithLBFGSAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, LogisticRegressionModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): LogisticRegressionModel = {
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    val lr = new LogisticRegressionWithLBFGS()
    lr.setNumClasses(ap.numClasses)
    lr.setIntercept(ap.intercept)
    lr.run(data.labeledPoints)
  }

  def predict(model: LogisticRegressionModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
