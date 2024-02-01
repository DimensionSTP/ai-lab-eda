# AI Lab EDA package

## AI Lab EDA package including machine learning and statistical analysis

### Dataset
Diabetes Health Indicators Dataset
👉🏻[Link](https://www.kaggle.com/datasets/julnazz/diabetes-health-indicators-dataset/data "Diabetes Health Indicators Dataset from kaggle")

### 🚀Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/ai-lab-eda.git
cd ai-lab-eda

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Before using package
Rename data files
diabetes_012_health_indicators_BRFSS2021.csv -> multiclasses.csv
diabetes_binary_5050split_health_indicators_BRFSS2021.csv -> undersampled.csv
diabetes_binary_health_indicators_BRFSS2021.csv -> binary.csv

Data files must be in a "data" folder

Download the libraries according to requirements.txt.

Modify the "project_dir" of the top config files to suit yours.

Sign up for wandb.ai and get an authentication token.

### Data split for machine learning(train / test)

* data preprocessing
```shell
python get_dataset.py
```

### Statistical Analysis(t-test, chi-squared)

* statistical_analysis for all dataset types
```shell
./scripts/analyze_statistics.sh
```

### Machine Learning(Diabetes classifier)

* model tuning
```shell
python main.py mode=tune
```

* model training
```shell
python main.py mode=train is_tuned={bool}
```

* model testing
```shell
python main.py mode=test
```

* all model tuning
```shell
./scripts/tune.sh
```

* all model training
```shell
./scripts/train.sh
```

* all model testing
```shell
./scripts/test.sh
```

__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__