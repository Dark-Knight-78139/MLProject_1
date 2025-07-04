import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessed_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        """
        Initialize the DataTransformationConfig with default paths.
        This method sets the paths for transformed training and testing data.
        """
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        """
        This function creates a data transformation pipeline.
        It handles both numerical and categorical features.
        """
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))  # StandardScaler does not support sparse matrices
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))  # StandardScaler does not support sparse matrices
            ])
            
            logging.info("Numerical and categorical pipelines created successfully")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function applies the data transformation pipeline to the training and testing datasets.
        It saves the transformed data to specified paths.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data loaded successfully for transformation")
            
            preprocessor_obj = self.get_data_transformation_object()
            
            target_column = 'math_score'
               
            X_train = train_df.drop(columns=[target_column],axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column],axis=1)
            y_test = test_df[target_column]
            
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)
            
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]
            
            logging.info("Data transformation completed successfully")
            
            save_object(
                file_path=self.data_transformation_config.preprocessed_object_file_path,
                obj=preprocessor_obj
            )
            
            return train_arr, test_arr, self.data_transformation_config.preprocessed_object_file_path
        except Exception as e:
            raise CustomException(e, sys)