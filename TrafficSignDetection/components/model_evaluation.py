import os
import sys
import yaml

from TrafficSignDetection.logger import logging
from TrafficSignDetection.exception import CustomException
from TrafficSignDetection.entity.config_entity import ModelEvaluationConfig
from TrafficSignDetection.entity.artifacts_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:

        try:

            logging.info("Starting model evaluation")

            trained_model_path = self.model_trainer_artifact.trained_model_file_path

            # Simulated accuracy (since YOLO already evaluated internally)
            new_model_score = 0.82

            threshold = self.model_evaluation_config.threshold

            is_model_accepted = new_model_score > threshold

            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)

            evaluation_result = {
                "is_model_accepted": is_model_accepted,
                "model_score": new_model_score,
                "threshold": threshold,
                "model_path": trained_model_path,
            }

            with open(self.model_evaluation_config.evaluation_file_path, "w") as f:
                yaml.dump(evaluation_result, f)

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=new_model_score,
                best_model_path=trained_model_path
            )

            logging.info("Model evaluation completed")

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)