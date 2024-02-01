HYDRA_FULL_ERROR=1 python main.py --multirun mode=tune metric=f1,accuracy tuning_way=original,cv
HYDRA_FULL_ERROR=1 python main.py --config-name=xgb.yaml --multirun mode=tune metric=f1,accuracy