import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_URL_KEY")
print(mongo_db_url)
import pymongo 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# Access database and collection via item access
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app=FastAPI()
origins=["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory="./templates")


@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.api_route("/train", methods=["GET", "POST"])
async def train_route(request: Request):
    try:
        payload = {}
        if request.method == "POST":
            try:
                payload = await request.json()
            except Exception:
                payload = {}

        # Browser fetch requests cannot include a body with GET/HEAD.
        # If you need to send data, use POST and send JSON.
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return {
            "status": "success",
            "message": "training is successful",
            "method": request.method,
            "payload": payload,
        }
    except Exception as e:
        raise NetworkSecurityException(e,sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        finals_model = load_object("finals_model/model.pkl")
        preprocesor = load_object("finals_model/preprocessor.pkl")
        network_model = NetworkModel(preprocessor=preprocesor, model=finals_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df["predicted_column"])
        os.makedirs("Prediction_output", exist_ok=True)
        df.to_csv("Prediction_output/output.csv", index=False)
        table_html = df.to_html(classes='table table-striped')
        # TemplateResponse expects (request, name, context)
        return templates.TemplateResponse(request, "table.html", {"table": table_html})

    except Exception as e:
        # Log the full exception and return a readable error response
        logging.exception("Prediction failed")
        try:
            err = NetworkSecurityException(e, sys)
            return Response(content=str(err), media_type="text/plain", status_code=500)
        except Exception:
            # Fallback to simple message if building the custom exception fails
            return Response(content=str(e), media_type="text/plain", status_code=500)
        
    


if __name__=="__main__":
    app_run(app,host="localhost",port=8000)
