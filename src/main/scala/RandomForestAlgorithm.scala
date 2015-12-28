package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class RFAlgorithmParams(
  numTrees: Int,
  featureSubsetStrategy: String,
  seed: Int
) extends Params

class RandomForestAlgorithm(val ap: RFAlgorithmParams)
  extends P2LAlgorithm[PreparedData, RandomForestModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): RandomForestModel = {
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    val treeStrategy = Strategy.defaultStrategy("Classification")
    RandomForest.trainClassifier(data.labeledPoints, treeStrategy, numTrees=ap.numTrees, 
      featureSubsetStrategy=ap.featureSubsetStrategy, seed = ap.seed)
  }

  def predict(model: RandomForestModel, query: Query): PredictedResult = {

    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label, Array(new Res(label,"random_forest",-1)))
  }

}
