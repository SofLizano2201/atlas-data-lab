"""Comparison coordinator; no AI decisions or duplicate metrics."""

from atlas.models.comparison import ExperimentComparison

compare = ExperimentComparison.compare


class ComparisonController:
    compare = staticmethod(ExperimentComparison.compare)
    rank = staticmethod(ExperimentComparison.rank)
    summary = staticmethod(ExperimentComparison.summary)
    groups = staticmethod(ExperimentComparison.groups)
    chart = staticmethod(ExperimentComparison.chart)

    @staticmethod
    def record_evaluation(state, experiments, metric, direction):
        from atlas.core.events import EventType, emit

        emit(
            state,
            EventType.COMPARISON_UPDATED,
            {
                "experiment_ids": [result.id for result in experiments],
                "metric": metric,
                "direction": direction,
            },
        )


from atlas.services.experiment_service import ExperimentService

record_experiment = ExperimentService.remember
