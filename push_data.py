import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where()
import pandas as pd
import numpy as np
import pymongo

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger

class NetworkDataExtract():
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.client['NetworkSecurity']
            logger.info("MongoDB connection established successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True)
            json_records = list(json.loads(df.T.to_json()).values())
            return json_records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data(self, json_records, database, collection_name):
        try:
            self.json_records = json_records
            self.database = self.client[database]
            self.collection_name = collection_name
            self.collection = self.database[self.collection_name]
            result = self.collection.insert_many(self.json_records)
            return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
if __name__ == "__main__":
    FILE_PATH = r"Network_Data\phisingData.csv"
    DATABASE = "AyushAI"
    COLLECTION_NAME = "NetworkData"
    networkobj=NetworkDataExtract()
    json_records=networkobj.cv_to_json(FILE_PATH)
    print(f"Number of records to be inserted: {len(json_records)}")
    no_of_records_inserted=networkobj.insert_data(json_records,DATABASE,COLLECTION_NAME)
    print(f"Number of records inserted: {no_of_records_inserted}")
    
