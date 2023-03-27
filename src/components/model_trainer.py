import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_mode_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and testing data")
            xtrain, ytrain, xtest, ytest =  (train_array[:, :-1], train_array[:,-1], 
                                            test_array[:, :-1], test_array[:,-1])

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XgbRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            logging.info("Training and testing data has been split")

            model_report: dict=evaluate_models(xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest, models=models)
            logging.info(f"Model report {model_report}")

            # Get the best model score
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score is {best_model_score}")
             
            # Get the best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            logging.info(f"Best model name is {best_model_name}")

            # Get the best model 
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")

            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_mode_file_path,
                obj=best_model
            )

            predicted_data = best_model.predict(xtest)
            logging.info("Test data has been predicted")

            r2score = r2_score(ytest, predicted_data)

            return r2score

        except Exception as e:
            raise CustomException(e, sys)



