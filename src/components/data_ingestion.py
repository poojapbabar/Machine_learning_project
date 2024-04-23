import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
import joblib
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    preprocessor_pkl_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    model_pkl_path: str = os.path.join('artifacts', 'model.pkl')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, target_column='target_column'):
        logging.info('Data Ingestion method started')
        try:
            csv_file_path = os.path.join('notebook','adult.csv')
            logging.debug('CSV file path:', csv_file_path)
            df = pd.read_csv(csv_file_path)
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved to file')

            logging.info('Train test split')
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

            logging.info('Preprocessing data')
            preprocessor = self._preprocess_data(X_train)

            logging.info('Training logistic regression model')
            model = self._train_model(X_train, y_train)

            logging.info('Saving preprocessor and model')
            self._save_preprocessor(preprocessor)
            self._save_model(model)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error('Exception occurred during data ingestion')
            raise CustomException(e, sys)
    def _save_preprocessor(self, preprocessor):
        try:
            joblib.dump(preprocessor, self.ingestion_config.preprocessor_pkl_path)
            logging.info('Preprocessor saved successfully')

            # Check if the file exists after saving
            if not os.path.exists(self.ingestion_config.preprocessor_pkl_path):
                logging.error('Error: Preprocessor file not created')
        except Exception as e:
            logging.error('Error occurred while saving preprocessor:', e)

    def _preprocess_data(self, X):
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        
        preprocessor.fit(X)
        return preprocessor

    def _train_model(self, X_train, y_train):
        model = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
        model.fit(X_train, y_train)
        return model

    def _save_preprocessor(self, preprocessor):
        joblib.dump(preprocessor, self.ingestion_config.preprocessor_pkl_path)

    def _save_model(self, model):
        joblib.dump(model, self.ingestion_config.model_pkl_path)
