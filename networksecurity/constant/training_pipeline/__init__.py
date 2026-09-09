import os
import sys
import numpy as np
import pandas as pd

"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN="Result"
PIPELINE_NAME:str="NetworkSecurity"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="phisingData.csv"

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH:str=os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME:str="NetworkData"
DATA_INGESTION_DATABASE_NAME:str="AyushAI"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingesteed"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:float=.2

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME:str= "data_validation"
DATA_VALIDATION_VALID_DIR:str= "validated"
DATA_VALIDATION_INVALID_DIR:str= "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str= "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str= "report.yaml"

"""
Data transformation related constant start with Data_TRANSFORMATIONVAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME:str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str="transfromed_object"

## knn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH:str="train.py"
DATA_TRANSFROMATION_TEST_FILE_PATH:str="test.py"
"""
Model Trainer related current start with mode trainer var name
"""
MODEL_TRAINER_DIR_NAME:str ="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str="model.pkl" 
MODEL_TRAINER_EXPECTED_SCORE:float=0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:float=.05
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD:float = MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
PREPROCESSING_OBJECT_FILE_NAME:str = "preprocessing.pkl"

# Backward-compatible exports for artifact classes that are defined in the entity layer.
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.exception.exception import NetworkSecurityException

__all__ = [
    "TARGET_COLUMN",
    "PIPELINE_NAME",
    "ARTIFACT_DIR",
    "FILE_NAME",
    "TRAIN_FILE_NAME",
    "TEST_FILE_NAME",
    "SCHEMA_FILE_PATH",
    "SAVED_MODEL_DIR",
    "MODEL_FILE_NAME",
    "DATA_INGESTION_COLLECTION_NAME",
    "DATA_INGESTION_DATABASE_NAME",
    "DATA_INGESTION_DIR_NAME",
    "DATA_INGESTION_FEATURE_STORE_DIR",
    "DATA_INGESTION_INGESTED_DIR",
    "DATA_INGESTION_TRAIN_TEST_SPLIT_RATION",
    "DATA_VALIDATION_DIR_NAME",
    "DATA_VALIDATION_VALID_DIR",
    "DATA_VALIDATION_INVALID_DIR",
    "DATA_VALIDATION_DRIFT_REPORT_DIR",
    "DATA_VALIDATION_DRIFT_REPORT_FILE_NAME",
    "DATA_TRANSFORMATION_DIR_NAME",
    "DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR",
    "DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR",
    "DATA_TRANSFORMATION_IMPUTER_PARAMS",
    "DATA_TRANSFORMATION_TRAIN_FILE_PATH",
    "DATA_TRANSFROMATION_TEST_FILE_PATH",
    "MODEL_TRAINER_DIR_NAME",
    "MODEL_TRAINER_TRAINED_MODEL_DIR",
    "MODEL_TRAINER_TRAINED_MODEL_NAME",
    "MODEL_TRAINER_EXPECTED_SCORE",
    "MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD",
    "MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD",
    "PREPROCESSING_OBJECT_FILE_NAME",
    "DataTransformationArtifact",
    "DataValidationArtifact",
    "NetworkSecurityException",
]

