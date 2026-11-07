
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
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
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define Hyperparameters
            params = {

                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "absolute_error"
                    ]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    "learning_rate": [.1, .01, .05, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    "n_neighbors": [5, 7, 9, 11]
                },

                "XGBoost Regressor": {
                    "learning_rate": [.1, .01, .05, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "iterations": [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [.1, .01, .5, .001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info(f"Model Report: {model_report}")

            # Find best model
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best R2 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final prediction
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
