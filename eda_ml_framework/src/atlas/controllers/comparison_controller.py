"""Comparison coordinator; no AI decisions or duplicate metrics."""

from atlas.models.comparison import ExperimentComparison

compare = ExperimentComparison.compare


class ComparisonController:
    compare = staticmethod(ExperimentComparison.compare)
    rank = staticmethod(ExperimentComparison.rank)
    summary = staticmethod(ExperimentComparison.summary)
    groups = staticmethod(ExperimentComparison.groups)
    chart = staticmethod(ExperimentComparison.chart)


from atlas.services.experiment_service import ExperimentService

record_experiment = ExperimentService.remember
