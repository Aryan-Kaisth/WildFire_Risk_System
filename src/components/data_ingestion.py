import os, sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import read_csv_file, save_csv_file

@dataclass
class DataIngestionConfig:
    """Holds file paths and parameters for the data ingestion process."""
    raw_data_dir: str = os.path.join("artifacts", "data_ingestion")
    raw_data_path: str = os.path.join("artifacts", "data_ingestion", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data_ingestion", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data_ingestion", "test.csv")

class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        """Initialize with a configuration object."""
        self.config = config
        os.makedirs(self.config.raw_data_dir, exist_ok=True)

    def initiate_data_ingestion(self, source_path: str):
        """Performs the entire data ingestion process."""
        logging.info("===== Data Ingestion Process Started =====")
        try:
            # Read raw data
            data = read_csv_file(source_path)
            logging.info(f"Data shape: {data.shape}")

            # Save raw copy
            save_csv_file(data, self.config.raw_data_path)
            logging.info(f"Raw data saved at {self.config.raw_data_path} with shape {data.shape}")

            # Step 3: Split train/test
            logging.info("Splitting data into train and test sets...")
            train_set, test_set = train_test_split(
                data,
                test_size=0.2,
                random_state=42
            )

            # Step 4: Save train/test sets
            save_csv_file(train_set, self.config.train_data_path)
            logging.info(f"Train data saved at {self.config.train_data_path} with shape {train_set.shape}")
            save_csv_file(test_set, self.config.test_data_path)
            logging.info(f"Test data saved at {self.config.test_data_path} with shape {test_set.shape}")

            logging.info("===== Data Ingestion Completed Successfully =====")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion.")
            raise CustomException(e, sys)
