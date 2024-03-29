from omegaconf import DictConfig
from hydra.utils import instantiate

from ..statistics.statistical_analysis import StatisticalAnalysis


class StatisticsSetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_statistical_analysis(self) -> StatisticalAnalysis:
        statistical_analysis: StatisticalAnalysis = instantiate(self.config.statistics)
        return statistical_analysis