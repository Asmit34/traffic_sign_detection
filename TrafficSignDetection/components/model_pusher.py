import os
import sys
import json
import shutil
from datetime import datetime

from TrafficSignDetection.logger import logging
from TrafficSignDetection.exception import CustomException

from TrafficSignDetection.entity.config_entity import ModelPusherConfig
from TrafficSignDetection.entity.artifacts_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)


class ModelPusher:
    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_evaluation_artifact: ModelEvaluationArtifact):

        self.config = model_pusher_config
        self.trainer_artifact = model_trainer_artifact
        self.eval_artifact = model_evaluation_artifact

    # ---------------------------
    # 📊 LOAD REGISTRY
    # ---------------------------
    def load_registry(self):
        if os.path.exists(self.config.registry_file_path):
            with open(self.config.registry_file_path, "r") as f:
                return json.load(f)
        return []

    # ---------------------------
    # 💾 SAVE REGISTRY
    # ---------------------------
    def save_registry(self, registry):
        with open(self.config.registry_file_path, "w") as f:
            json.dump(registry, f, indent=4)

    # ---------------------------
    # 🔢 GET NEXT VERSION
    # ---------------------------
    def get_next_version(self, registry):
        if not registry:
            return "v1"
        last_version = registry[-1]["version"]
        version_num = int(last_version.replace("v", "")) + 1
        return f"v{version_num}"

    # ---------------------------
    # 🚀 PUSH MODEL
    # ---------------------------
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("🚀 Starting Model Pusher")

            if not self.eval_artifact.is_model_accepted:
                raise Exception("Model not accepted. Skipping push.")

            os.makedirs(self.config.saved_model_dir, exist_ok=True)

            registry = self.load_registry()

            # 🔍 Get best previous score
            if registry:
                best_score = max([m["score"] for m in registry])
            else:
                best_score = 0

            new_score = self.eval_artifact.model_score

            logging.info(f"Previous best score: {best_score}")
            logging.info(f"New model score: {new_score}")

            # ❌ If not better → skip
            if new_score <= best_score:
                logging.info("New model is NOT better → Skipping push")
                return ModelPusherArtifact(
                    pushed_model_path=None,
                    model_version="not_pushed"
                )

            # ✅ PUSH MODEL
            version = self.get_next_version(registry)

            version_dir = os.path.join(self.config.saved_model_dir, version)
            os.makedirs(version_dir, exist_ok=True)

            source_model = self.trainer_artifact.trained_model_file_path
            destination_model = os.path.join(version_dir, "best.pt")

            shutil.copy(source_model, destination_model)

            # 📝 Update registry
            model_entry = {
                "version": version,
                "score": new_score,
                "path": destination_model,
                "timestamp": str(datetime.now())
            }

            registry.append(model_entry)
            self.save_registry(registry)

            logging.info(f"✅ Model pushed as {version}")

            return ModelPusherArtifact(
                pushed_model_path=destination_model,
                model_version=version
            )

        except Exception as e:
            raise CustomException(e, sys)