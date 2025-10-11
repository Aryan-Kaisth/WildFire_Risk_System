import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_numpy_array_data, save_object, read_csv_file, read_yaml_file

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
    transformed_train_file_path: str = os.path.join("artifacts", "data_transformation", "train.npy")
    transformed_test_file_path: str = os.path.join("artifacts", "data_transformation", "test.npy")

class DataTransformation:
    SCHEMA_PATH = os.path.join("config", "schema.yaml")

    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            # Read schema using your utility function
            self.schema = read_yaml_file(self.SCHEMA_PATH)
            self.drop_cols = self.schema.get("drop_columns")
            self.transformation_cols = self.schema.get("transform_columns", [])
            self.num_cols = self.schema.get("numerical_columns", [])
            self.target_column = self.schema.get("target_column")
            logging.info(f"Schema loaded successfully. Drop: {self.drop_cols}, Transform: {self.transformation_cols}, Numerical: {self.num_cols}, Target: {self.target_column}")
        except Exception as e:
            logging.error("Error initializing DataTransformation with schema")
            raise CustomException(e, sys)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.drop_duplicates(ignore_index=True, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['quarter'] = df['datetime'].dt.quarter
            df['dayofyear'] = df['datetime'].dt.dayofyear
            df['weekofyear'] = df['datetime'].dt.isocalendar().week.astype(int)
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            df['trange'] = df['tmmx'] - df['tmmn']
            df['rrange'] = df['rmax'] - df['rmin']
            df['fm_ratio'] = df['fm100'] / df['fm1000']
            df['pet_minus_etr'] = df['pet'] - df['etr']
            df['trange_srad'] = df['trange'] * df['srad']
            df['vpd_tmmx'] = df['vpd'] * df['tmmx']
            df['fm_wind'] = df['fm100'] * df['vs']
            df['pr_rmax_ratio'] = df['pr'] / df['rmax']
            df['fm_diff'] = df['fm100'] - df['fm1000']
            df.drop(columns=self.drop_cols, inplace=True, errors='ignore')
            df[self.target_column] = df[self.target_column].map({"No": 0, "Yes": 1})
            return df
        except Exception as e:
            logging.error("Error in feature engineering")
            raise CustomException(e, sys)

    def get_preprocessor_pipeline(self):
        try:
            preprocessor = Pipeline([
                ('yeojohnson', YeoJohnsonTransformer(variables=self.transformation_cols)),
                ('winsorizer', Winsorizer(capping_method='quantiles', tail='both', fold="auto")),
                ('scaler', StandardScaler())
            ])
            return preprocessor
        except Exception as e:
            logging.error("Error creating preprocessor Pipeline")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading train and test data")
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            logging.info("Applying feature engineering on train and test data")
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            X_train = train_df.drop(columns=[self.target_column], axis=1)
            y_train = train_df[self.target_column]
            X_test = test_df.drop(columns=[self.target_column], axis=1)
            y_test = test_df[self.target_column]

            logging.info("Creating preprocessing Pipeline")
            preprocessor = self.get_preprocessor_pipeline()

            logging.info("Fitting preprocessor on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("Saving transformed data and preprocessor")
            save_numpy_array_data(self.config.transformed_train_file_path, np.c_[X_train_transformed, y_train])
            save_numpy_array_data(self.config.transformed_test_file_path, np.c_[X_test_transformed, y_test])
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            
            logging.info(f"Data transformation completed and saved successfully at {self.config.preprocessor_obj_file_path}")
            return X_train_transformed, X_test_transformed, y_train, y_test
        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)
