HYDRA_FULL_ERROR=1 python main.py --multirun mode=tune metric=f1,accuracy tuning_way=original,cv
HYDRA_FULL_ERROR=1 python main.py --multirun --config-name=xgb.yaml mode=tune metric=f1,accuracy