import hydra

from src.pipeline.statistics_pipeline import analyze_statistics


@hydra.main(config_path="configs/", config_name="analyze_statistics.yaml")
def main(config):
    return analyze_statistics(config)


if __name__ == "__main__":
    main()