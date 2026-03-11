import sys, os
from TrafficSignDetection.logger import logging
from TrafficSignDetection.exception import CustomException
from TrafficSignDetection.components.data_ingestion import DataIngestion
from TrafficSignDetection.components.data_validation import DataValidation
from TrafficSignDetection.components.model_trainer import ModelTrainer
from TrafficSignDetection.components.model_evaluation import ModelEvaluation

from TrafficSignDetection.entity.config_entity import (DataIngestionConfig,
                                                 DataValidationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig
                                                 )

from TrafficSignDetection.entity.artifacts_entity import (DataIngestionArtifact,
                                                    DataValidationArtifact,
                                                    ModelTrainerArtifact,
                                                    ModelEvaluationArtifact)
                                                    


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()



    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try: 
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )
            logging.info("Getting the data from URL")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from URL")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)
        


    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_trainer(self) -> ModelTrainerArtifact:
        try:

            trained_model_path = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "yolov5_training_results",
                "weights",
                "best.pt"
            )

            # ✅ Check if model already exists
            if os.path.exists(trained_model_path):

                logging.info("Model already exists. Skipping training.")

                model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=trained_model_path
                )

                return model_trainer_artifact

            # 🚀 Otherwise train model
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_evaluation(
        self,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:

        try:

            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)


    def run_pipeline(self) -> None:
            try:
                data_ingestion_artifact = self.start_data_ingestion()

                data_validation_artifact = self.start_data_validation(
                    data_ingestion_artifact=data_ingestion_artifact
                )

                if data_validation_artifact.validation_status:

                    model_trainer_artifact = self.start_model_trainer()

                    model_evaluation_artifact = self.start_model_evaluation(
                        model_trainer_artifact=model_trainer_artifact
                    )

                else:
                    raise Exception("Your data is not in correct format")

            except Exception as e:
                raise CustomException(e, sys)
        
