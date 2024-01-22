from omegaconf import DictConfig
import hydra

from src.pipeline.ml_pipeline import train


@hydra.main(config_path="configs/", config_name="diabetes_lgbm_classifier_train.yaml")
def main(config: DictConfig,) -> None:
    return train(config)


if __name__ == "__main__":
    main()