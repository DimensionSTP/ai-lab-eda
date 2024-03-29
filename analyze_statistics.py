from omegaconf import DictConfig
import hydra

from src.pipelines.statistics_pipeline import analyze_statistics


@hydra.main(config_path="configs/", config_name="analyze_statistics.yaml")
def main(config: DictConfig,) -> None:
    return analyze_statistics(config)


if __name__ == "__main__":
    main()