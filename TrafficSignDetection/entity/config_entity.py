import os
from dataclasses import dataclass
from TrafficSignDetection.constant import training_pipeline 


@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = training_pipeline.ARTIFACTS_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig() 

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, training_pipeline.DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR
    )

    data_download_url: str = training_pipeline.DATA_DOWNLOAD_URL
