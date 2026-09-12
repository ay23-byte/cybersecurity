import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_transformer_object(self) -> Pipeline:
        try:
            return Pipeline(
                steps=[
                    ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS))
                ]
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    from networksecurity.utils.ml_utils.mlflow_utils import capture_run_logs

    @capture_run_logs
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the validated dataset.")

            train_features = train_df.drop(columns=[TARGET_COLUMN])
            train_target = train_df[TARGET_COLUMN]
            test_features = test_df.drop(columns=[TARGET_COLUMN])
            test_target = test_df[TARGET_COLUMN]

            preprocessor = self.get_transformer_object()
            transformed_train_data = preprocessor.fit_transform(train_features)
            transformed_test_data = preprocessor.transform(test_features)

            train_array = np.c_[transformed_train_data, train_target.to_numpy()]
            test_array = np.c_[transformed_test_data, test_target.to_numpy()]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            save_object(self.data_transformation_config.transformed_object_file_path, obj=preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_array)

            save_object("finals_model/preprocessor.pkl",preprocessor)
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact created: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e