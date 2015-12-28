package org.template.classification

import io.prediction.controller.EngineFactory
import io.prediction.controller.Engine

class Query(
  val features: Array[Double]
) extends Serializable

class PredictedResult(
  val label: Double,
  val rset: Array[Res]
) extends Serializable

class Res(
  val label: Double,
  val alg: String,
  val prob: Double
) extends Serializable

class ActualResult(
  val label: Double
) extends Serializable

object ClassificationEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map(
        "naive" -> classOf[NaiveBayesAlgorithm],
        "random_forest" -> classOf[RandomForestAlgorithm],
	      "log_reg_lbfg" -> classOf[LRWithLBFGSAlgorithm]
      ),
      classOf[Serving])
  } 
}
