import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(eval_metric='logloss'),
                'CatBoostRegressor': CatBoostRegressor(verbose=False)
            }
            
            params = {
                "RandomForestRegressor": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "DecisionTreeRegressor": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "GradientBoostingRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LinearRegression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoostRegressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            model_report : dict=evaluate_model(X_train, y_train, X_test, y_test, models = models, param = params)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                obj=best_model_name, 
                file_path=self.model_trainer_config.trained_model_file_path
            )
            
            pred = models[best_model_name].predict(X_test)
            r2_square = r2_score(y_test, pred)
            return r2_square
                   
        except Exception as e:
            raise CustomException(e, sys)

