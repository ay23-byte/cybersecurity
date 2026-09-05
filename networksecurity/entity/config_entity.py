from datetime import datetime
import os
from networksecurity.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)


class TrainingPipelineConfig:
    def __init__(self,time_stamp=datetime.now()):
        time_stamp=datetime.now().strftime("%m%d%Y__%H%M%S")
        self.pipeline_name=training_pipeline.PIPELINE_NAME
        self.artifact_name=training_pipeline.ARTIFACT_DIR
        self.artifact_dir=os.path.join(self.artifact_name,time_stamp)
        self.time_stamp: str=time_stamp


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
      self.data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
      self.feature_store_dir:str=os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
      self.feature_store_file_path:str=os.path.join(self.feature_store_dir,training_pipeline.FILE_NAME)
      self.train_file_path:str=os.path.join(self.data_ingestion_dir,training_pipeline.TRAIN_FILE_NAME)
      self.test_file_path:str=os.path.join(self.data_ingestion_dir,training_pipeline.TEST_FILE_NAME)
      # Backward-compatible aliases for older code paths.
      self.training_file_path:str=self.train_file_path
      self.testing_file_path:str=self.test_file_path
      self.train_test_split_ratio:float=training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
      self.collection_name:str=training_pipeline.DATA_INGESTION_COLLECTION_NAME
      self.database_name:str=training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir:str=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir:str=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path:str=os.path.join(self.valid_data_dir,training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path:str=os.path.join(self.valid_data_dir,training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path:str=os.path.join(self.invalid_data_dir,training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path:str=os.path.join(self.invalid_data_dir,training_pipeline.TEST_FILE_NAME)
        self.drift_report_dir:str=os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)
        self.drift_report_file_path:str=os.path.join(self.drift_report_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
