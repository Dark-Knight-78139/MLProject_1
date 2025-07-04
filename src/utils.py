import os
import sys

import pandas as pd
import numpy as np

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