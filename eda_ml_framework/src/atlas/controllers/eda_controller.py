"""Explicit workflow adapter; scientific calculations remain in models."""

from typing import Any

import pandas as pd

from atlas.models import ExploratoryDataAnalysis


class EDAController:
    model_type = ExploratoryDataAnalysis

    def profile(self) -> dict[str, Any]:
        return self._model.profile()

    def transformation_impact(self, candidate: pd.DataFrame) -> dict[str, Any]:
        return self._model.transformation_impact(candidate)

    def outlier_diagnostic(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._model.outlier_diagnostic(*args, **kwargs)

    def __init__(
        self, source: ExploratoryDataAnalysis | pd.DataFrame, random_state: int = 42
    ) -> None:
        self._model = (
            source
            if isinstance(source, ExploratoryDataAnalysis)
            else self.model_type(source, random_state=random_state)
        )

    @property
    def random_state(self) -> int:
        return self._model.random_state

    @property
    def original(self) -> Any:
        return self._model.original

    @property
    def dataset_version(self) -> Any:
        return self._model.dataset_version

    @property
    def df(self) -> Any:
        return self._model.df

    @property
    def history(self) -> Any:
        return self._model.history

    @property
    def version_count(self) -> Any:
        return self._model.version_count

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(cls.model_type.load(*args, **kwargs))

    @classmethod
    def load_csv(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(cls.model_type.load_csv(*args, **kwargs))

    @classmethod
    def load_excel(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(cls.model_type.load_excel(*args, **kwargs))

    @classmethod
    def load_json(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(cls.model_type.load_json(*args, **kwargs))

    @classmethod
    def load_parquet(cls, *args: Any, **kwargs: Any) -> Any:
        return cls(cls.model_type.load_parquet(*args, **kwargs))

    def preview(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.preview(*args, **kwargs)

    def dimensions(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.dimensions(*args, **kwargs)

    def columns(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.columns(*args, **kwargs)

    def data_types(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.data_types(*args, **kwargs)

    def memory_usage(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.memory_usage(*args, **kwargs)

    def information(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.information(*args, **kwargs)

    def unique_values(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.unique_values(*args, **kwargs)

    def constant_columns(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.constant_columns(*args, **kwargs)

    def duplicates(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.duplicates(*args, **kwargs)

    def backup(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.backup(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.apply(*args, **kwargs)

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.restore(*args, **kwargs)

    def export(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.export(*args, **kwargs)

    def statistics(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.statistics(*args, **kwargs)

    def statistic(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.statistic(*args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.mean(*args, **kwargs)

    def median(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.median(*args, **kwargs)

    def mode(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.mode(*args, **kwargs)

    def minimum(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.minimum(*args, **kwargs)

    def maximum(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.maximum(*args, **kwargs)

    def value_range(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.value_range(*args, **kwargs)

    def variance(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.variance(*args, **kwargs)

    def standard_deviation(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.standard_deviation(*args, **kwargs)

    def quartiles(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.quartiles(*args, **kwargs)

    def interquartile_range(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.interquartile_range(*args, **kwargs)

    def coefficient_of_variation(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.coefficient_of_variation(*args, **kwargs)

    def skewness(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.skewness(*args, **kwargs)

    def kurtosis(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.kurtosis(*args, **kwargs)

    def percentiles(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.percentiles(*args, **kwargs)

    def frequencies(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.frequencies(*args, **kwargs)

    def contingency(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.contingency(*args, **kwargs)

    def summary(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.summary(*args, **kwargs)

    def grouped_statistics(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.grouped_statistics(*args, **kwargs)

    def missing_values(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.missing_values(*args, **kwargs)

    def outliers(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.outliers(*args, **kwargs)

    def type_inconsistencies(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.type_inconsistencies(*args, **kwargs)

    def rare_categories(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.rare_categories(*args, **kwargs)

    def transform(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.transform(*args, **kwargs)

    def custom_transform(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.custom_transform(*args, **kwargs)

    def correlation(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.correlation(*args, **kwargs)

    def vif(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.vif(*args, **kwargs)

    def statistical_test(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.statistical_test(*args, **kwargs)

    def chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.chart(*args, **kwargs)

    def report(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.report(*args, **kwargs)
