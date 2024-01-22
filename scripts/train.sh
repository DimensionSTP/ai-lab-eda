HYDRA_FULL_ERROR=1 python train.py
HYDRA_FULL_ERROR=1 python train.py data_type=binary_imbalanced
HYDRA_FULL_ERROR=1 python train.py data_type=undersampled
HYDRA_FULL_ERROR=1 python train.py --config-name=diabetes_xgb_classifier_train.yaml
HYDRA_FULL_ERROR=1 python train.py --config-name=diabetes_xgb_classifier_train.yaml data_type=binary_imbalanced
HYDRA_FULL_ERROR=1 python train.py --config-name=diabetes_xgb_classifier_train.yaml data_type=undersampled