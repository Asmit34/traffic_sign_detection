import os
import sys
import yaml
import zipfile
from TrafficSignDetection.main_utils import read_yaml_file
from TrafficSignDetection.logger import logging
from TrafficSignDetection.exception import CustomException
from TrafficSignDetection.entity.config_entity import ModelTrainerConfig
from TrafficSignDetection.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            def unzip_data(zip_file_path):
                logging.info("Unzipping data")

                # Get the current working directory
                cwd = os.getcwd()

                # Extract the zip file to the current working directory
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(cwd)

                # Remove the zip file
                os.remove(zip_file_path)

            # Example usage
            unzip_data("traffic.zip")

            # Read data.yaml (no modification of 'nc' here)
            with open("dataset.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])


            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            # Read the existing model configuration
            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            # Run training script
            os.system(f"cd yolov5/ && python train.py --img 415 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../dataset.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results --cache")

            # Copy the best model file to the yolov5 directory
            os.system("copy yolov5\\runs\\train\\yolov5s_results\\weights\\best.pt yolov5\\")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"copy yolov5\\runs\\train\\yolov5s_results\\weights\\best.pt {self.model_trainer_config.model_trainer_dir}\\")


            # Clean up
            os.system("rmdir /s /q yolov5\\runs")
            os.system("rmdir /s /q train")
            os.system("rmdir /s /q valid")


            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
