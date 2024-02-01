import os
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy import stats


class StatisticalAnalysis():
    def __init__(
        self,
        df_path: str,
        features: List[str],
        method: str,
        result_path: str,
        result_name: str,
    ) -> None:
        self.df = pd.read_csv(df_path)
        self.features = features
        self.method = method
        self.result_path = result_path
        self.result_name = result_name

    def __call__(self) -> None:
        positive, negative = self.get_groups()
        self.analyze_statistics(positive, negative)

    def get_groups(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        positive = self.df[(self.df["Diabetes_binary"] == 1)]
        negative = self.df[(self.df["Diabetes_binary"] == 0)]
        return (positive, negative)

    def analyze_statistics(
        self,
        positive: pd.DataFrame,
        negative: pd.DataFrame,
    ) -> None:
        num_positive = len(positive)
        num_negative = len(negative)

        t_values = []
        p_values = []
        meaningful_features = []
        if self.method == "t_test":
            for feature in self.features:
                positive_values = positive[feature].values
                negative_values = negative[feature].values
                t_value, p_value = stats.ttest_ind(positive_values, negative_values)
                t_values.append(t_value)
                p_values.append(p_value)

            for i, p_value in enumerate(p_values):
                if p_value < 0.05:
                    meaningful_features.append(self.features[i])
        elif self.method == "chi2":
            for feature in self.features:
                contingency_table = pd.crosstab(self.df[feature], self.df["Diabetes_binary"])
                chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)
                p_values.append(p_value)

            for i, p_value in enumerate(p_values):
                if p_value < 0.05:
                    meaningful_features.append(self.features[i])
        else:
            raise ValueError("Invalid method")

        result = {
            "분석 방법": self.method,
            "유의한 지표": meaningful_features,
            "당뇨 샘플 수": num_positive,
            "비당뇨 샘플 수": num_negative,
        }
        result_df = pd.DataFrame.from_dict(result, orient="index").T

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path, exist_ok=True)

        result_file = f"{self.result_path}/{self.result_name}"
        if os.path.isfile(result_file):
            original_result_df = pd.read_csv(result_file)
            new_result_df = pd.concat([original_result_df, result_df], ignore_index=True)
            new_result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
        else:
            result_df.to_csv(
                result_file,
                encoding="utf-8-sig",
                index=False,
            )
