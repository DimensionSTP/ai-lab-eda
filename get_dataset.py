import pandas as pd
from tqdm import tqdm


def get_dataset(
    data_path: str,
    data_name: str,
    fix_num_diabetes: int,
    save_name: str,
) -> None:
    df = pd.read_csv(f"{data_path}/{data_name}")
    non_diabetes = df[df["Diabetes_binary"] == 0]
    diabetes = df[df["Diabetes_binary"] == 1]

    if fix_num_diabetes != 0:
        non_diabetes_sample = non_diabetes.sample(n=fix_num_diabetes, random_state=2024)
        diabetes_sample = diabetes.sample(n=fix_num_diabetes, random_state=2024)
    else:
        non_diabetes_sample = non_diabetes.sample(frac=0.1, random_state=2024)
        diabetes_sample = diabetes.sample(frac=0.1, random_state=2024)

    test_data = pd.concat([non_diabetes_sample, diabetes_sample])
    train_data = df.drop(test_data.index)
    test_data.to_csv(f"{data_path}/{save_name}_test.csv", index=False)
    train_data.to_csv(f"{data_path}/{save_name}_train.csv", index=False)


if __name__ == "__main__":
    DATA_PATH = "./data"
    DATA_NAMES = ["binary.csv", "binary.csv", "undersampled.csv"]
    FIX_NUM_DIABETES_OPTION = [7000, 0, 0]
    SAVE_NAMES = ["binary_balanced", "binary_imbalanced", "undersampled"]

    for i in tqdm(range(3)):
        get_dataset(
            data_path=DATA_PATH,
            data_name=DATA_NAMES[i],
            fix_num_diabetes=FIX_NUM_DIABETES_OPTION[i],
            save_name=SAVE_NAMES[i],
        )