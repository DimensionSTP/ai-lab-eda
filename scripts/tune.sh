HYDRA_FULL_ERROR=1 python tune.py --multirun metric=f1,accuracy tuning_way=original,cv
HYDRA_FULL_ERROR=1 python tune.py --config-name=diabetes_xgb_classifier_tune.yaml
HYDRA_FULL_ERROR=1 python tune.py --config-name=diabetes_xgb_classifier_tune.yaml metric=accuracy