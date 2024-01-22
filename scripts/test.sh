HYDRA_FULL_ERROR=1 python test.py
HYDRA_FULL_ERROR=1 python test.py data_type=binary_imbalanced
HYDRA_FULL_ERROR=1 python test.py data_type=undersampled
HYDRA_FULL_ERROR=1 python test.py --config-name=diabetes_xgb_classifier_test.yaml
HYDRA_FULL_ERROR=1 python test.py --config-name=diabetes_xgb_classifier_test.yaml data_type=binary_imbalanced
HYDRA_FULL_ERROR=1 python test.py --config-name=diabetes_xgb_classifier_test.yaml data_type=undersampled