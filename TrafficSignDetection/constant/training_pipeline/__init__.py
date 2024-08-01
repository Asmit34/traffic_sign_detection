ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME

"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1KUfKeq9WmCf3-R9hlIwZoTFTYcyNsN3T/view?usp=drive_link"

"""s
Data Validation related constant start with DATA_Validation VAR NAME

"""
DATA_VALIDATION_DIR_NAME : str = "data_valdation"
DATA_VALIDATION_STATUS_FILE = "status.txt"
DATA_VALIDATION_ALL_REQUIRED_FILE = ["train", "val", "dataset.yaml"]

"""
Model trainer related constant start with DATA_INGESTION VAR NAME

"""
MODEL_TRIANER_DIR_NAME :str = "model_trainer"
MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov5s.pt"
MODEL_TRIANER_NO_EPOCHS:int = 50
MODEL_TRAINER_BATCH_SIZE: int  = 16
