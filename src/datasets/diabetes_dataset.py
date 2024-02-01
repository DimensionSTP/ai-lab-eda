from typing import List, Tuple

import pandas as pd


class DiabetesDataset():
    def __init__(
        self,
        df_path: str,
        meaningful_features: List[str],
    ) -> None:
        self.df_path = df_path
        self.meaningful_features = meaningful_features

    def __call__(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = self.get_diabetes_dataset()
        return dataset

    def get_diabetes_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.df_path)
        data = df[self.meaningful_features]
        label = df["Diabetes_binary"]
        return (data, label)