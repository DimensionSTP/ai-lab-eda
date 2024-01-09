from omegaconf import DictConfig
from hydra.utils import instantiate

from ..dataset_modules.dataset import DiabetesDataset
from ..architecture_modules.basic_archimodule import BasicClassifierModule


class MLSetUp:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_diabetes_dataset(self) -> DiabetesDataset:
        diabetes_dataset: DiabetesDataset = instantiate(self.config.dataset_module)
        return diabetes_dataset

    def get_basic_classifier(self) -> BasicClassifierModule:
        basic_archimodule: BasicClassifierModule = instantiate(self.config.architecture_module)
        return basic_archimodule