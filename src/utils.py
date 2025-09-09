import os
import sys
import numpy as np
import pandas as pd
from src.exception import customException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb" ) as file:
            dill.dump(obj,file)

    except Exception as e:
        raise customException(e)  


def evaluateModel(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            search = RandomizedSearchCV(
                                        estimator=model,
                                        param_distributions=para,
                                        n_iter=10,        # only 10 random combos
                                        cv=5,
                                        scoring='r2',
                                        n_jobs=1,
                                        random_state=42
                                    )
            search.fit(x_train,y_train)
            
            model.set_params(**search.best_params_)
            model.fit(x_train,y_train)
            

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report    
    except Exception as e:
        raise customException(e)