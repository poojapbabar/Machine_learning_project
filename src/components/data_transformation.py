import sys
from dataclasses import dataclass
import os
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_function

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define categorical and numerical columns
            categorical_cols = ['workclass', 'marital-status', 'occupation', 'native-country']
            numerical_cols = ['educational-num', 'age']

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder())
            ])

            # ColumnTransformer to apply pipelines to respective columns
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            # Target column and columns to remove
            target_column_name = 'income'
            remove_col = ['fnlwgt', 'education', 'relationship', 'race', 'capital-gain', 'capital-loss', 'hours-per-week']
            drop_columns = [target_column_name, remove_col]

            # Splitting features and target
            X_train = train_df.drop(columns=drop_columns, axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=drop_columns, axis=1)
            y_test = test_df[target_column_name]
            
            # Transforming data
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Saving preprocessor object
            save_function(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            logging.info('Preprocessor pickle file saved')

            return X_train_transformed, X_test_transformed, y_train, y_test, self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)
