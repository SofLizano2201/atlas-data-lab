"""Dataset lifecycle coordinator."""

from atlas.services.dataset_service import DatasetService

from .eda_controller import EDAController


class DatasetController(EDAController):
    activate = staticmethod(DatasetService.activate)
    example = staticmethod(DatasetService.example)
    upgrade = staticmethod(DatasetService.upgrade)
