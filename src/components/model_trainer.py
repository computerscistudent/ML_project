from src.logger import logging
import os
from src.utils import save_obj
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from src.exception import customException
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from src.utils import save_obj,evaluateModel

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data.")
            x_train,y_train,x_test,y_test = ( train_arr[:,:-1],
                                             train_arr[:,-1],
                                             test_arr[:,:-1],
                                             test_arr[:,-1] )
            
            models = {
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "adaBoost": AdaBoostRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoostRegressor": CatBoostRegressor(),
                    "GradientBoosting": GradientBoostingRegressor()
                }

            model_report = evaluateModel(x_train,y_train,x_test,y_test,models)

            best_model_scores = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_scores)]

            best_model = models[best_model_name]

            if best_model_scores < 0.6:
                raise customException(Exception("no best model found"))
            
            logging.info("Best model found for both training and testing datasets.")



        except Exception as e:
            raise customException(e)    



