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

            logging.info("Training and testing data has been split")

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
            
            params = {
                "Random Forest": 
                {
                    "n_estimators": [40, 60, 80, 100, 120], 
                    "max_depth": [5, 8, 10], 
                    "min_samples_leaf": [2, 4, 6, 8]
                },

                "Decision Tree":
                {
                    "criterion": ['squared_error', 'absolute_error'],
                    "min_samples_leaf": [1, 2, 3]
                },

                "Gradient Boosting": 
                {
                    "loss": ['squared_error', 'absolute_error'],
                    "n_estimators": [50, 80, 100, 120]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": 
                {
                    "n_neighbors": [3, 4, 6, 8, 10]
                },

                "XgbRegressor": 
                {
                    "n_estimators": [50, 80, 100, 120],
                    "max_depth": [4, 6, 8, 10],
                    "eta": [0.3, 0.1, 0.01]
                },

                "CatBoosting Regressor": {},
                
                "AdaBoost Regressor": 
                {
                    "n_estimators": [50, 80, 100, 120],
                    "learning_rate": [0.03, 0.05, 0.1, 0.001]
                }
            }


            model_report: dict=evaluate_models(xtrain=xtrain, xtest=xtest, ytrain=ytrain, ytest=ytest, 
                                                models=models, params=params)
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



