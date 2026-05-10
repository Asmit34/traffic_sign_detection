from TrafficSignDetection.pipeline.training_pipeline import TrainPipeline
import sys
if __name__ == "__main__":
    force_train = "--force_train" in sys.argv

    pipeline = TrainPipeline()
    pipeline.force_train = force_train

    pipeline.run_pipeline()
