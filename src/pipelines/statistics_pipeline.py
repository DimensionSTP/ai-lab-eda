from omegaconf import DictConfig

from ..utils.statistics_setup import StatisticsSetUp


def analyze_statistics(config: DictConfig,) -> None:
    statistics_setup = StatisticsSetUp(config)

    statistical_analysis = statistics_setup.get_statistical_analysis()
    statistical_analysis()