import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(xtrain, xtest, ytrain, ytest, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, scoring='neg_mean_squared_error', cv=3)
            gs.fit(xtrain, ytrain)
            
            best_params = gs.best_params_
            logging.info(f"Best parameters for {model} is {best_params}")

            model.set_params(**best_params) # set the best parameters into the model

            model.fit(xtrain, ytrain) # train model

            ytrain_pred = model.predict(xtrain)

            ytest_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain, ytrain_pred)

            test_model_score = r2_score(ytest, ytest_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
