from TrafficSignDetection.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    # Create an instance of TrainPipeline
    pipeline = TrainPipeline()
    
    # Start the pipeline
    pipeline.run_pipeline()
