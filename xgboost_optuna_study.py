import os
from typing import Tuple
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
import optuna

from matplotlib import pyplot as plt


class End2EndStudy():
    def __init__(
        self,
        data_path: str,
        test_size: float,
        seed: int,
        study_direction: str,
        save_path: str,
        num_trials: int,
    ):
        self.data_path = data_path
        self.test_size = test_size
        self.seed = seed
        self.study_direction = study_direction
        self.num_trials = num_trials
        self.save_path = f"{save_path}/{num_trials}_trials"
        self.x_train, self.x_test, self.y_train, self.y_test = self.get_dataset()

    def __call__(self) -> None:
        study=optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_trials)
        trial = study.best_trial
        print(f"Accuracy : {trial.value}")
        print(f"Parameters : {trial.params}")

        model = xgb.XGBClassifier(**trial.params, random_state=self.seed)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print(f"Test Accuracy : {accuracy}")
        print(f"Test Report : {report}")
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        with open(f"{self.save_path}/best_params.json", "w") as json_file:
            json.dump(trial.params, json_file)

        model.save_model(f"{self.save_path}/xgboost_model.model")

        with open(f"{self.save_path}/classification_report.txt", "w") as file:
            file.write(report)

        xgb.plot_importance(model)
        plt.savefig(f"{self.save_path}/importance_features.png")
        plt.show()

    def get_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        multiclasses_df = pd.read_csv(f"{self.data_path}/multiclasses.csv")
        binary_df = pd.read_csv(f"{self.data_path}/binary.csv")
        undersampled_df = pd.read_csv(f"{self.data_path}/undersampled.csv")
        multiclasses_df["Diabetes_012"] = multiclasses_df["Diabetes_012"].replace({1: 1, 2: 1, 0: 0})
        multiclasses_df.rename(columns={"Diabetes_012": "Diabetes_binary"}, inplace=True)
        df = pd.concat([multiclasses_df, binary_df, undersampled_df], ignore_index=True)
        x = df.drop(columns=["Diabetes_binary"])
        y = df["Diabetes_binary"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.seed)
        return (x_train, x_test, y_train, y_test)

    def objective(self, trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "eta": trial.suggest_loguniform("eta", 1e-8, 1.0),
            "gamma": trial.suggest_loguniform("gamma", 1e-8, 1.0),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        }

        model = xgb.XGBClassifier(**params)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy


if __name__ == "__main__":
    DATA_PATH = "./data"
    TEST_SIZE = 0.2
    SEED = 2024
    DIRECTION = "maximize"
    NUM_TRIALS = 100
    SAVE_PATH = "./result"
    study = End2EndStudy(
        data_path=DATA_PATH,
        test_size=TEST_SIZE,
        seed=SEED,
        study_direction=DIRECTION,
        num_trials=NUM_TRIALS,
        save_path=SAVE_PATH,
    )
    study()