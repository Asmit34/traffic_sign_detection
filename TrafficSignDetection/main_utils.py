import os.path
import sys
import yaml
import base64
from TrafficSignDetection.exception import CustomException
from TrafficSignDetection.logger import logging

def read_yaml_file(file_path:str)-> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)
 
    except Exception as e:
        raise CustomException(e, sys)
    
def decodeImage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open("./data" + filename, "wb") as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:

      return base64.b64encode(f.read())
