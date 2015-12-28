package org.template.classification

import io.prediction.controller.LServing

class Serving extends LServing[Query, PredictedResult] {

  override
  def serve(query: Query,
    predictedResults: Seq[PredictedResult]): PredictedResult = {
    val rset = predictedResults.map(x=>(x.rset(0))).toArray

    new PredictedResult(rset(0).label, rset)
  }
}
