from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

## Configuration for data ingestion

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
ca = certifi.where()

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) 


    def export_collection_as_dataframe(self):
        """
        Read datafrom MongoDB
        """
        try:
           if not MONGO_DB_URL:
               raise ValueError("MONGO_DB_URL is not set in the environment")
           database_name = self.data_ingestion_config.database_name
           collection_name = self.data_ingestion_config.collection_name
           self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
           collection = self.mongo_client[database_name][collection_name]
           df = pd.DataFrame(list(collection.find()))

           if "_id" in df.columns.to_list():
               df= df.drop("_id", axis=1)

           df.replace(to_replace="na", value=np.nan, inplace=True)
           return df
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

    def export_data_to_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Create folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys) 

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Performed train test split on the dataset")
            logging.info(
                "Existing data ingestion component. Train test split is done and train and test file is created."
            )
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting training dataset to file: [{self.data_ingestion_config.train_file_path}]")

            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(f"Exporting test dataset to file: [{self.data_ingestion_config.test_file_path}]")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def initiate_data_ingestion(self):
        try:
           dataframe=self.export_collection_as_dataframe()
           dataframe=self.export_data_to_feature_store(dataframe)
           self.split_data_as_train_test(dataframe)
           self.data_ingestion_artifact=DataIngestionArtifact(train_file_path=self.data_ingestion_config.train_file_path,test_file_path=self.data_ingestion_config.test_file_path)
           return self.data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) 
