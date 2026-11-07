
import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

    try:
        report = {}

        for model_name, model in models.items():

            logging_message = f"Training {model_name}"
            print(logging_message)

            # Get parameters safely
            para = param.get(model_name, {})

            # Hyperparameter tuning
            if para:

                gs = GridSearchCV(
                    model,
                    para,
                    cv=3,
                    scoring="r2",
                    error_score="raise"
                )

                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            # Evaluation
            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            print(f"{model_name}")
            print(f"Train R2 Score: {train_model_score}")
            print(f"Test R2 Score: {test_model_score}")
            print("-" * 50)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
