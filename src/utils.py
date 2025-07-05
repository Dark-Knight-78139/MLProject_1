import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(obj, file_path):
    """
    This function saves an object to a specified file path.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            import joblib
            joblib.dump(obj, file)
    except Exception as e:
        raise Exception(f"Error saving object: {str(e)}")
    
def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    This function evaluates multiple models and returns their performance metrics.
    """
    
    try :
        model_report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            
            gs.fit(X_train, y_train)
            # model.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            model_report[list(models.keys())[i]] = r2_square
            
        return model_report
    except Exception as e:
        raise CustomException(e, sys)