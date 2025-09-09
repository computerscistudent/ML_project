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
                    "DecisionTreeRegressor": DecisionTreeRegressor(),
                    "RandomForestRegressor": RandomForestRegressor(),
                    "AdaBoostRegressor": AdaBoostRegressor(),
                    "LinearRegression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoostRegressor": CatBoostRegressor(verbose=0),
                    "GradientBoostingRegressor": GradientBoostingRegressor()
                }

            params = {
                    "DecisionTreeRegressor": [
                        {
                            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                            "splitter": ["best", "random"],
                            "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4]
                        }
                    ],

                    "RandomForestRegressor": [
                        {
                            "n_estimators": [50, 100, 200],
                            "criterion": ["squared_error", "absolute_error"],
                            "max_depth": [None, 5, 10, 20],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2", None]
                        }
                    ],

                    "AdaBoostRegressor": [
                        {
                            "n_estimators": [50, 100, 200],
                            "learning_rate": [0.01, 0.1, 0.5, 1.0],
                            "loss": ["linear", "square", "exponential"]
                        }
                    ],

                    "LinearRegression": [
                        {
                            "fit_intercept": [True, False],
                            "positive": [True, False]
                        }
                    ],

                    "XGBRegressor": [
                        {
                            "n_estimators": [100, 200, 500],
                            "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [3, 5, 7, 10],
                            "subsample": [0.6, 0.8, 1.0],
                            "colsample_bytree": [0.6, 0.8, 1.0],
                            "reg_alpha": [0, 0.01, 0.1],
                            "reg_lambda": [1, 1.5, 2]
                        }
                    ],

                    "CatBoostRegressor": [
                        {
                            "iterations": [200, 500, 1000],
                            "depth": [4, 6, 8, 10],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "l2_leaf_reg": [1, 3, 5, 7, 9],
                            "border_count": [32, 64, 128]
                        }
                    ],

                    "GradientBoostingRegressor": [
                        {
                            "n_estimators": [100, 200, 500],
                            "learning_rate": [0.01, 0.05, 0.1, 0.2],
                            "max_depth": [3, 5, 7],
                            "min_samples_split": [2, 5, 10],
                            "min_samples_leaf": [1, 2, 4],
                            "subsample": [0.6, 0.8, 1.0],
                            "max_features": ["sqrt", "log2", None]
                        }
                    ]
                }




            model_report = evaluateModel(x_train,y_train,x_test,y_test,models,params)

            best_model_scores = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_scores)]

            best_model = models[best_model_name]

            if best_model_scores < 0.6:
                raise customException(Exception("no best model found"))
            
            logging.info("Best model found for both training and testing datasets.")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(x_test)

            score = r2_score(y_test,predicted)

            return score



        except Exception as e:
            raise customException(e)    



