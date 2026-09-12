import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import  load_numpy_array_data, save_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression# type: ignore
from sklearn.neighbors import KNeighborsClassifier# type: ignore
from sklearn.tree import DecisionTreeClassifier# type: ignore
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)# type: ignore
import dagshub# type: ignore
dagshub.init(repo_owner='ay23-byte', repo_name='cybersecurity', mlflow=True)
import mlflow# type: ignore
import tempfile
import contextlib
import os

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation: DataTransformationArtifact = None,
                 data_transformation_artifact: DataTransformationArtifact = None):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation or data_transformation_artifact
            if self.data_transformation_artifact is None:
                raise ValueError("Missing data transformation artifact.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self,best_model, classificationmetric):
        try:
            # Prepare a temporary file to capture stdout/stderr during the MLflow run
            tmp = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log", prefix="mlflow_run_")
            tmp.close()
            try:
                with mlflow.start_run():
                    # Redirect stdout/stderr into the temp log file so we can save run logs as artifacts
                    with open(tmp.name, "a", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                        # Safely extract metric values
                        f1_score = getattr(classificationmetric, "f1_score", None)
                        precision_score = getattr(classificationmetric, "precision_score", None)
                        recall_score = getattr(classificationmetric, "recall_score", None)

                        if f1_score is not None:
                            try:
                                mlflow.log_metric("f1_score", float(f1_score))
                            except Exception as e:
                                logging.error(f"Failed to log f1_score: {e}")
                                mlflow.set_tag("mlflow.log_metric_error_f1", str(e))

                        if precision_score is not None:
                            try:
                                mlflow.log_metric("precision_score", float(precision_score))
                            except Exception as e:
                                logging.error(f"Failed to log precision_score: {e}")
                                mlflow.set_tag("mlflow.log_metric_error_precision", str(e))

                        if recall_score is not None:
                            try:
                                mlflow.log_metric("recall_score", float(recall_score))
                            except Exception as e:
                                logging.error(f"Failed to log recall_score: {e}")
                                mlflow.set_tag("mlflow.log_metric_error_recall", str(e))

                        # Log the model; failures here shouldn't crash the run
                        try:
                            mlflow.sklearn.log_model(best_model, "model")
                        except Exception as e:
                            logging.error(f"Failed to log model to MLflow: {e}")
                            mlflow.set_tag("mlflow.log_model_error", str(e))

                    # After closing redirects, upload the captured run log as an artifact
                    try:
                        mlflow.log_artifact(tmp.name, artifact_path="run_logs")
                    except Exception as e:
                        logging.error(f"Failed to upload run log artifact: {e}")
                        mlflow.set_tag("mlflow.log_artifact_error", str(e))
            finally:
                # Remove the temporary local file; MLflow copied it if upload succeeded
                try:
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Unexpected error during MLflow tracking: {e}")

    def train_model(self, X_train, y_train, X_test, y_test):

        print("\n=== Model Training Started ===")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        models = {
                'LogisticRegression': LogisticRegression(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier()
            }

        params={
            "DecisionTreeClassifier": {
                'criterion': ['gini', 'entropy', 'log_loss']
            },
            "RandomForestClassifier": {
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },

            "GradientBoostingClassifier": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, .7, .75, .8, .85, .9],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },
            "LogisticRegression": {},
            "AdaBoostClassifier": {
                'learning_rate': [.1, .01, .05, .001],
                "n_estimators": [8, 16, 32, 64, 128, 256]

            }
        } 

        print("\n=== Hyperparameter Tuning Started ===")
        model_report:dict =evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

        best_model_score=max(sorted(model_report.values())) 

        ## to get best model name from dict
        best_model_name=list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        print(f"Best model by F1 score: {best_model_name} with score {best_model_score}")

        best_model=models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        ## Track theexperiments with  mlflow
        self.track_mlflow(best_model, classification_train_metric)




        y_test_pred=best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)
        print(f"Train metrics: {classification_train_metric}")
        print(f"Test metrics: {classification_test_metric}")

        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
        print(f"Saved trained model to: {self.model_trainer_config.trained_model_file_path}")

        save_object("finals_model/model.pkl",best_model)
        
        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )

        logging.info(f"Model trainer artifact:{model_trainer_artifact}")
        return model_trainer_artifact




    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
   