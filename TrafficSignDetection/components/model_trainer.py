import os
import sys
import yaml
import torch
import shutil
import subprocess
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
            # def unzip_data(zip_file_path):
            #     logging.info("Unzipping data")
            #     cwd = os.getcwd()

            #     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            #         zip_ref.extractall(cwd)

            #     os.remove(zip_file_path)

            # unzip_data("traffic.zip")

            # Read dataset.yaml
            with open("dataset.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            custom_model_path = f'yolov5/models/custom_{model_config_file_name}.yaml'
            with open(custom_model_path, 'w') as f:
                yaml.dump(config, f)

            # ---------------------------
            # 🚀 TRAIN YOLOv5
            # ---------------------------
            train_command = [
                "python",
                "train.py",
                "--img", "416",
                "--device", "0" if torch.cuda.is_available() else "cpu",
                "--batch", str(self.model_trainer_config.batch_size),
                "--epochs", str(self.model_trainer_config.no_epochs),
                "--data", "../dataset.yaml",
                "--cfg", f"./models/custom_{model_config_file_name}.yaml",
                "--weights", self.model_trainer_config.weight_name,
                "--name", "yolov5s_results",
                "--cache"
            ]
            # Log the command for debugging
            logging.info(f"Running YOLOv5 training command: {' '.join(train_command)}")

            # Run subprocess
            subprocess.run(train_command, cwd="yolov5", check=True)
            # ---------------------------
            # 📁 COPY FULL TRAINING RESULTS
            # ---------------------------
            source_dir = os.path.join("yolov5", "runs", "train", "yolov5s_results")

            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            destination_dir = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "yolov5_training_results"
            )

            # Copy entire folder (NOT just best.pt)
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

            # Path to best.pt inside copied folder
            trained_model_path = os.path.join(
                destination_dir,
                "weights",
                "best.pt"
            )

            # ---------------------------
            # RETURN ARTIFACT
            # ---------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_path
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
