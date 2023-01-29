import logging
from logging.config import dictConfig

import uvicorn
from fastapi import FastAPI

from config import AppConfig, LogConfig
from data_containers import PredictRequestSchema, PredictResponseSchema, UpdateResponseSchema
from model_wrapper import Pipeline

dictConfig(LogConfig().dict())
logger = logging.getLogger("qa_bert")

app = FastAPI()
app_config = AppConfig()

pipeline = Pipeline()


@app.post("/predict")
def predict(input_data: PredictRequestSchema) -> PredictResponseSchema:
    """
    Endpoint to get predictions from BERT QA.

    :param input_data: input data with context and question
    :type input_data: PredictRequestSchema
    :return: answer
    :rtype: PredictResponseSchema
    """
    try:
        logger.debug("Prediction process is started")
        prediction = pipeline.predict(input_data)
        response = PredictResponseSchema(answer=prediction)
        logger.debug("Prediction process finished successfully")
    except Exception as e:
        response = PredictResponseSchema(error=str(e))
        logger.error(f"Prediction process finished with error: {e}")
    return response


@app.post("/update")
def update() -> None:
    """
    Endpoint to reload trained model.

    :return: None
    """
    logger.debug("Model reload was requested")
    try:
        pipeline.load_artifacts()
        response = UpdateResponseSchema()
        logger.info("Detector was reloaded")
    except Exception as e:
        response = UpdateResponseSchema(error=str(e))
        logger.error(f"Prediction process finished with error: {e}")
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=app_config.host, port=app_config.port)
