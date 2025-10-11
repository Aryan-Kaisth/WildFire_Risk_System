# src/components/model_trainer.py
import os, sys
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model_trainer", "histgbm.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        # Make sure directory exists
        os.makedirs(os.path.dirname(self.config.model_file_path), exist_ok=True)

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Trains a Histogram Gradient Boosting Classifier model and evaluates it.
        Saves the trained model as a pickle file.
        Returns the accuracy and classification report.
        """
        try:
            logging.info("Model training started")
            # Initialize model
            histgbm = HistGradientBoostingClassifier(max_iter=100, random_state=42, class_weight="balanced")

            # Train model
            histgbm.fit(X_train, y_train)
            logging.info("Model training completed")

            # Predict on test set
            y_pred = histgbm.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            logging.info(f"Model Accuracy: {accuracy}")
            logging.info(f"Precision Score:\n{precision}")
            logging.info(f"Recall Score:\n{recall}")
            logging.info(f"Classification Report:\n{report}")

            # Save model
            save_object(self.config.model_file_path, histgbm)
            logging.info(f"Trained model saved at: {self.config.model_file_path}")

            return histgbm

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
