"""Scientific model implementation, independent of agents and Streamlit."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
)

from atlas.core.compatibility import normalize_transform_parameters
from atlas.core.exceptions import FrameworkError
from atlas.core.schemas import DatasetVersion

from .experiment import export_frame, html_report
from .preprocessing import SCALERS
from .provenance import fingerprint


class ExploratoryDataAnalysis:
    """Encapsulate immutable source data and versioned exploratory transformations."""

    @property
    def source_columns(self) -> list[str]:
        """Return source schema names without copying or exposing original rows."""
        return list(self._original.columns)

    def outlier_diagnostic(
        self, columns: list[str], method: str = "iqr", threshold: float = 1.5
    ) -> dict[str, Any]:
        """Summarize the existing outlier detector once, with a bounded row preview."""
        mask = self.outliers(columns, method, threshold).any(axis=1)
        return {"count": int(mask.sum()), "rows": self._df[mask].head(100).copy()}

    def profile(self) -> dict[str, Any]:
        """Compose existing data-quality outputs for controllers and agent context."""
        rows, columns = self.dimensions()
        return {
            "rows": rows,
            "columns": columns,
            "missing_cells": int(self.missing_values()["cantidad"].sum()),
            "missing_percentage": float(self._df.isna().to_numpy().mean() * 100),
            "additional_duplicates": int(self._df.duplicated().sum()),
            "constant_columns": len(self.constant_columns()),
            "numeric_columns": len(self._df.select_dtypes(include=np.number).columns),
            "categorical_columns": len(self._df.select_dtypes(exclude=np.number).columns),
            "temporal_columns": len(
                self._df.select_dtypes(include=["datetime", "datetimetz"]).columns
            ),
        }

    def feature_roles(self) -> dict[str, list[str]]:
        """Flag explicit identifier/date names conservatively; retain continuous unique values."""
        identifiers, dates = [], []
        for column in self._df:
            name = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
            if name in {"id", "index", "uuid", "codigo", "identificador"} or name.endswith(
                ("_id", "_uuid")
            ):
                identifiers.append(column)
            if (
                name in {"date", "fecha", "timestamp", "datetime"}
                or name.endswith(("_date", "_fecha", "_timestamp"))
                or pd.api.types.is_datetime64_any_dtype(self._df[column])
            ):
                dates.append(column)
        constants = self.constant_columns()
        excluded = set(identifiers + dates + constants)
        return {
            "identifiers": identifiers,
            "dates": dates,
            "constants": constants,
            "suggested": [column for column in self._df if column not in excluded],
        }

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        self._validate(df)
        self._df = df.copy(deep=True)
        self._original = df.copy(deep=True)
        self._dataset_id = fingerprint(self._original)
        self._content_hash = self._dataset_id
        self.random_state = random_state
        self._versions = [self._df.copy(deep=True)]
        self._history: list[dict[str, Any]] = []

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise FrameworkError("Se requiere un DataFrame con filas y columnas.")
        if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
            raise FrameworkError("Las columnas deben tener nombres únicos de texto.")
        if len(df) > 100000 or len(df.columns) > 500:
            raise FrameworkError("Límite interactivo: 100 000 filas y 500 columnas.")
        if df.memory_usage(deep=True).sum() > 100 * 1024**2:
            raise FrameworkError("El DataFrame supera 100 MB en memoria.")

    @property
    def original(self) -> pd.DataFrame:
        """Return a defensive copy of the immutable original."""
        return self._original.copy(deep=True)

    @property
    def dataset_version(self) -> DatasetVersion:
        """Describe the original identity and current content fingerprint."""
        applied = [h for h in self._history if h.get("estado") == "aplicada"]
        return DatasetVersion(
            dataset_id=self._dataset_id,
            version=len(self._versions) - 1,
            content_hash=self._content_hash,
            rows=len(self._df),
            columns=self.columns(),
            operation=str(applied[-1]["operacion"]) if applied else "carga",
        )

    @property
    def df(self) -> pd.DataFrame:
        """Return a defensive copy of the active dataset."""
        return self._df.copy(deep=True)

    @property
    def history(self) -> pd.DataFrame:
        """Return a copy of the transformation audit trail."""
        return pd.DataFrame(self._history)

    @property
    def version_count(self) -> int:
        """Provide version count using validated inputs."""
        return len(self._versions)

    @classmethod
    def load(
        cls,
        source: Any,
        file_format: str | None = None,
        separator: str = ",",
        text_encoding: str = "utf-8",
        sheet: str | int = 0,
    ) -> ExploratoryDataAnalysis:
        """Read a validated tabular source with size, format and encoding checks."""
        name = getattr(source, "name", str(source))
        fmt = (file_format or Path(name).suffix.lstrip(".")).lower()
        if fmt not in {"csv", "xlsx", "json", "parquet"}:
            raise FrameworkError("Use CSV, XLSX, JSON o Parquet.")
        if isinstance(source, (str, Path)):
            if Path(source).stat().st_size > 25 * 1024**2:
                raise FrameworkError("El archivo supera 25 MB.")
            data = Path(source).read_bytes()
        elif isinstance(source, bytes):
            data = source
        else:
            source.seek(0)
            data = source.read(25 * 1024**2 + 1)
        if len(data) > 25 * 1024**2:
            raise FrameworkError("El archivo supera 25 MB.")
        try:
            buffer = BytesIO(data)
            if fmt == "csv":
                if len(separator) != 1:
                    raise FrameworkError("El separador debe ser un carácter.")
                frame = pd.read_csv(buffer, sep=separator, encoding=text_encoding, nrows=100001)
            elif fmt == "xlsx":
                from zipfile import ZipFile

                with ZipFile(buffer) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 100 * 1024**2:
                        raise FrameworkError("Excel descomprimido supera 100 MB.")
                buffer.seek(0)
                frame = pd.read_excel(buffer, sheet_name=sheet, nrows=100001)
            elif fmt == "json":
                frame = pd.read_json(buffer)
            else:
                import pyarrow.parquet as pq

                metadata = pq.ParquetFile(buffer).metadata
                if (
                    metadata.num_rows > 100000
                    or sum(
                        metadata.row_group(i).total_byte_size
                        for i in range(metadata.num_row_groups)
                    )
                    > 100 * 1024**2
                ):
                    raise FrameworkError("Parquet excede límites de filas o memoria.")
                buffer.seek(0)
                frame = pd.read_parquet(buffer)
            if frame.memory_usage(deep=True).sum() > 100 * 1024**2:
                raise FrameworkError("Dataset descomprimido supera 100 MB.")
            return cls(frame)
        except (ValueError, UnicodeError, OSError, KeyError, BadZipFile) as exc:
            raise FrameworkError(f"No se pudo leer el archivo: {exc}") from exc

    @classmethod
    def load_csv(cls, source: Any, **config: Any) -> ExploratoryDataAnalysis:
        """Provide load csv using validated inputs."""
        return cls.load(source, file_format="csv", **config)

    @classmethod
    def load_excel(cls, source: Any, sheet: str | int = 0) -> ExploratoryDataAnalysis:
        """Provide load excel using validated inputs."""
        return cls.load(source, file_format="xlsx", sheet=sheet)

    @classmethod
    def load_json(cls, source: Any) -> ExploratoryDataAnalysis:
        """Provide load json using validated inputs."""
        return cls.load(source, file_format="json")

    @classmethod
    def load_parquet(cls, source: Any) -> ExploratoryDataAnalysis:
        """Provide load parquet using validated inputs."""
        return cls.load(source, file_format="parquet")

    def preview(self, n: int = 10) -> pd.DataFrame:
        """Provide preview using validated inputs."""
        return self.df.head(n)

    def dimensions(self) -> tuple[int, int]:
        """Provide dimensions using validated inputs."""
        return self._df.shape

    def visualization_sample(self, maximum: int = 1000):
        """Return a reproducible display-only model without replacing source statistics."""
        if maximum < 1:
            raise FrameworkError("Chart point limit must be positive")
        frame = self._df.sample(min(maximum, len(self._df)), random_state=self.random_state)
        return type(self)(frame, random_state=self.random_state)

    def columns(self) -> list[str]:
        """Provide columns using validated inputs."""
        return list(self._df.columns)

    def data_types(self) -> pd.Series:
        """Provide data types using validated inputs."""
        return self._df.dtypes.astype(str)

    def memory_usage(self) -> pd.Series:
        """Provide memory usage using validated inputs."""
        return self._df.memory_usage(deep=True)

    def information(self) -> pd.DataFrame:
        """Provide information using validated inputs."""
        return pd.DataFrame(
            {
                "tipo": self.data_types(),
                "no_nulos": self._df.count(),
                "unicos": self._df.nunique(),
                "faltantes_%": self.missing_values().porcentaje,
                "bytes": self.memory_usage().drop("Index", errors="ignore"),
            }
        )

    def unique_values(self, column: str) -> np.ndarray:
        """Provide unique values using validated inputs."""
        self._columns([column])
        return self._df[column].unique()

    def constant_columns(self) -> list[str]:
        """Provide constant columns using validated inputs."""
        return list(self._df.columns[self._df.nunique(dropna=False) <= 1])

    def duplicates(self) -> pd.DataFrame:
        """Provide duplicates using validated inputs."""
        return self.df[self._df.duplicated(keep=False)]

    def backup(self) -> int:
        """Provide backup using validated inputs."""
        if len(self._versions) >= 20:
            raise FrameworkError("Máximo de 20 versiones; exporte y cree una nueva sesión.")
        self._validate_version_memory(self._df)
        self._versions.append(self.df)
        return len(self._versions) - 1

    def _validate_version_memory(self, frame: pd.DataFrame) -> None:
        """Provide validate version memory using validated inputs."""
        used = sum(v.memory_usage(deep=True).sum() for v in self._versions)
        if used + frame.memory_usage(deep=True).sum() > 256 * 1024**2:
            raise FrameworkError(
                "Historial de versiones limitado a 256 MB; exporte antes de continuar."
            )

    def transformation_impact(self, candidate: pd.DataFrame) -> dict[str, Any]:
        """Describe a preview without changing either frame or fitting a model."""
        self._validate(candidate)
        changed = [
            column
            for column in self._df
            if column not in candidate or not self._df[column].equals(candidate[column])
        ]
        return {
            "rows_before": len(self._df),
            "rows_after": len(candidate),
            "columns_before": len(self._df.columns),
            "columns_after": len(candidate.columns),
            "missing_before": int(self._df.isna().to_numpy().sum()),
            "missing_after": int(candidate.isna().to_numpy().sum()),
            "changed_columns": changed,
            "added_columns": [column for column in candidate if column not in self._df],
        }

    def apply(self, new_frame: pd.DataFrame, operation: str, confirm: bool = False) -> None:
        """Apply a validated proposal only after explicit confirmation and record its version."""
        if not confirm:
            raise FrameworkError("Confirme explícitamente la modificación.")
        self._validate(new_frame)
        if len(self._versions) >= 20:
            raise FrameworkError("Límite de versiones alcanzado.")
        self._validate_version_memory(new_frame)
        new_hash = fingerprint(new_frame)
        self._df = new_frame.copy(deep=True)
        self._content_hash = new_hash
        version = self.backup()
        self._history.append({"operacion": operation, "estado": "aplicada", "version": version})

    def restore(self, version: int, confirm: bool = False) -> None:
        """Restore a prior dataset version only after explicit confirmation."""
        if not 0 <= version < len(self._versions):
            raise FrameworkError("Versión inexistente.")
        self.apply(self._versions[version], f"restaurar {version}", confirm)

    def export(self, file_format: str = "csv") -> bytes:
        """Provide export using validated inputs."""
        return export_frame(self._df, file_format)

    def _columns(self, columns: list[str], numeric_only: bool = False) -> None:
        if (
            not columns
            or len(set(columns)) != len(columns)
            or any(c not in self._df for c in columns)
        ):
            raise FrameworkError("Seleccione columnas existentes y sin repeticiones.")
        if numeric_only and any(not pd.api.types.is_numeric_dtype(self._df[c]) for c in columns):
            raise FrameworkError("La operación requiere columnas numéricas.")

    def statistics(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Calculate numeric descriptive statistics while handling missing values."""
        columns = (
            columns
            if columns is not None
            else list(self._df.select_dtypes(include=np.number).columns)
        )
        self._columns(columns, True)
        x = self._df[columns]
        result = pd.DataFrame(
            {
                "media": x.mean(),
                "mediana": x.median(),
                "moda": x.apply(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
                "minimo": x.min(),
                "maximo": x.max(),
                "rango": x.max() - x.min(),
                "varianza": x.var(),
                "desviacion": x.std(),
                "q1": x.quantile(0.25),
                "q2": x.quantile(0.5),
                "q3": x.quantile(0.75),
                "iqr": x.quantile(0.75) - x.quantile(0.25),
                "cv": x.std() / x.mean().abs().replace(0, np.nan),
                "asimetria": x.skew(),
                "curtosis": x.kurt(),
                "n": x.count(),
            }
        )
        return result

    def statistic(self, name: str, column: str) -> float:
        """Provide statistic using validated inputs."""
        table = self.statistics([column])
        if name not in table:
            raise FrameworkError("Estadística desconocida.")
        return float(table.loc[column, name])

    def mean(self, column: str) -> float:
        """Provide mean using validated inputs."""
        return self.statistic("media", column)

    def median(self, column: str) -> float:
        """Provide median using validated inputs."""
        return self.statistic("mediana", column)

    def mode(self, column: str) -> pd.Series:
        """Provide mode using validated inputs."""
        self._columns([column])
        return self._df[column].mode()

    def minimum(self, column: str) -> float:
        """Provide minimum using validated inputs."""
        return self.statistic("minimo", column)

    def maximum(self, column: str) -> float:
        """Provide maximum using validated inputs."""
        return self.statistic("maximo", column)

    def value_range(self, column: str) -> float:
        """Provide value range using validated inputs."""
        return self.statistic("rango", column)

    def variance(self, column: str) -> float:
        """Provide variance using validated inputs."""
        return self.statistic("varianza", column)

    def standard_deviation(self, column: str) -> float:
        """Provide standard deviation using validated inputs."""
        return self.statistic("desviacion", column)

    def quartiles(self, column: str) -> pd.Series:
        """Provide quartiles using validated inputs."""
        return self.percentiles(column, [25, 50, 75])

    def interquartile_range(self, column: str) -> float:
        """Provide interquartile range using validated inputs."""
        return self.statistic("iqr", column)

    def coefficient_of_variation(self, column: str) -> float:
        """Provide coefficient of variation using validated inputs."""
        return self.statistic("cv", column)

    def skewness(self, column: str) -> float:
        """Provide skewness using validated inputs."""
        return self.statistic("asimetria", column)

    def kurtosis(self, column: str) -> float:
        """Provide kurtosis using validated inputs."""
        return self.statistic("curtosis", column)

    def percentiles(self, column: str, values: list[float]) -> pd.Series:
        """Provide percentiles using validated inputs."""
        self._columns([column], True)
        if any(not 0 <= v <= 100 for v in values):
            raise FrameworkError("Percentiles fuera de [0,100].")
        return self._df[column].quantile(np.asarray(values) / 100)

    def frequencies(self, column: str) -> pd.DataFrame:
        """Provide frequencies using validated inputs."""
        self._columns([column])
        n = self._df[column].value_counts(dropna=False)
        return pd.DataFrame({"absoluta": n, "relativa": n / len(self._df)})

    def contingency(self, a: str, b: str) -> pd.DataFrame:
        """Provide contingency using validated inputs."""
        self._columns([a, b])
        if self._df[a].nunique(dropna=False) * self._df[b].nunique(dropna=False) > 10000:
            raise FrameworkError("Contingencia limitada a 10000 celdas; agrupe categorías primero.")
        return pd.crosstab(self._df[a].fillna("(faltante)"), self._df[b].fillna("(faltante)"))

    def summary(self) -> pd.DataFrame:
        """Provide summary using validated inputs."""
        return self._df.describe(include="all").T

    def grouped_statistics(self, category: str, column: str) -> pd.DataFrame:
        """Provide grouped statistics using validated inputs."""
        self._columns([category, column])
        self._columns([column], True)
        return self._df.groupby(category, dropna=False, observed=True)[column].describe()

    def missing_values(self) -> pd.DataFrame:
        """Provide missing values using validated inputs."""
        n = self._df.isna().sum()
        return pd.DataFrame({"cantidad": n, "porcentaje": 100 * n / len(self._df)})

    def outliers(
        self, columns: list[str], method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """Provide outliers using validated inputs."""
        self._columns(columns, True)
        if threshold <= 0:
            raise FrameworkError("El umbral debe ser positivo.")
        x = self._df[columns]
        if method == "iqr":
            q1, q3 = (x.quantile(0.25), x.quantile(0.75))
            return (x < q1 - threshold * (q3 - q1)) | (x > q3 + threshold * (q3 - q1))
        if method == "zscore":
            return ((x - x.mean()) / x.std(ddof=0).replace(0, np.nan)).abs() > threshold
        if method == "isolation":
            if x.isna().all().any():
                raise FrameworkError("Impute columnas completamente vacías primero.")
            mask = (
                IsolationForest(random_state=self.random_state).fit_predict(x.fillna(x.median()))
                == -1
            )
            return pd.DataFrame({c: mask for c in columns}, index=x.index)
        raise FrameworkError("Método de outliers desconocido.")

    def type_inconsistencies(self) -> pd.DataFrame:
        """Provide type inconsistencies using validated inputs."""
        return pd.DataFrame(
            [
                {
                    "columna": c,
                    "tipos": len({type(v).__name__ for v in self._df[c].dropna()}),
                    "numericos_en_texto": int(
                        pd.to_numeric(self._df[c], errors="coerce").notna().sum()
                    ),
                }
                for c in self._df.select_dtypes(exclude=np.number).columns
            ]
        )

    def rare_categories(self, column: str, threshold: float = 0.01) -> pd.DataFrame:
        """Provide rare categories using validated inputs."""
        if not 0 <= threshold <= 1:
            raise FrameworkError("Umbral entre 0 y 1.")
        table = self.frequencies(column)
        return table[table.relativa < threshold]

    def transform(
        self, operation: str, columns: list[str] | None = None, **params: Any
    ) -> pd.DataFrame:
        """Return a transformation proposal without mutating the active dataset."""
        params = normalize_transform_parameters(params)
        columns = columns if columns is not None else self.columns()
        self._columns(columns)
        out = self.df
        x = out[columns]
        numeric_ops = {
            *SCALERS,
            "mean",
            "median",
            "knn",
            "log",
            "boxcox",
            "yeojohnson",
            "binning",
            "clip",
            "winsor",
            "drop_outliers",
        }
        if operation in numeric_ops:
            self._columns(columns, True)
        if operation == "drop_rows":
            out = out.dropna(subset=columns)
        elif operation == "drop_null_columns":
            threshold = float(params.get("threshold", 0.5))
            if not 0 <= threshold <= 1:
                raise FrameworkError("Umbral entre 0 y 1.")
            out = out.drop(columns=[c for c in columns if out[c].isna().mean() > threshold])
        elif operation == "duplicates":
            out = out.drop_duplicates(subset=columns)
        elif operation in {"mean", "median", "mode", "constant"}:
            for c in columns:
                value = (
                    x[c].mean()
                    if operation == "mean"
                    else x[c].median()
                    if operation == "median"
                    else params.get("value", 0)
                    if operation == "constant"
                    else x[c].mode().iloc[0]
                    if not x[c].mode().empty
                    else np.nan
                )
                if pd.isna(value):
                    raise FrameworkError(f"{c}: no hay valores para imputar.")
                out[c] = out[c].fillna(value)
        elif operation == "knn":
            if len(x) > 5000:
                raise FrameworkError("Imputación KNN interactiva limitada a 5000 filas.")
            out[columns] = KNNImputer(
                n_neighbors=int(params.get("n_neighbors", 5)), keep_empty_features=True
            ).fit_transform(x)
        elif operation in SCALERS:
            out[columns] = SCALERS[operation]().fit_transform(x)
        elif operation in {"onehot", "ordinal"}:
            if (
                operation == "onehot"
                and sum(x[c].nunique(dropna=False) for c in columns) * len(x) > 10000000
            ):
                raise FrameworkError(
                    "One-hot excedería 10 millones de celdas; excluya identificadores."
                )
            encoder = (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                if operation == "onehot"
                else OrdinalEncoder()
            )
            encoded = encoder.fit_transform(x.astype(str))
            names = encoder.get_feature_names_out(columns)
            if set(names) & set(out.columns) - set(columns):
                raise FrameworkError("La codificación produciría nombres duplicados.")
            out = pd.concat(
                [out.drop(columns=columns), pd.DataFrame(encoded, columns=names, index=out.index)],
                axis=1,
            )
        elif operation == "label":
            if len(columns) != 1 or x.isna().any().any():
                raise FrameworkError("LabelEncoder requiere una sola variable objetivo sin nulos.")
            out[columns[0]] = LabelEncoder().fit_transform(x.iloc[:, 0])
        elif operation == "log":
            if (x <= 0).any().any():
                raise FrameworkError("Log requiere valores estrictamente positivos.")
            out[columns] = np.log(x)
        elif operation in {"boxcox", "yeojohnson"}:
            out[columns] = PowerTransformer(
                method="box-cox" if operation == "boxcox" else "yeo-johnson"
            ).fit_transform(x)
        elif operation == "binning":
            out[columns] = KBinsDiscretizer(
                n_bins=int(params.get("n_bins", 5)),
                encode="ordinal",
                strategy="quantile",
                subsample=None,
                quantile_method="averaged_inverted_cdf",
            ).fit_transform(x)
        elif operation == "select":
            out = out[columns]
        elif operation == "drop_columns":
            out = out.drop(columns=columns)
        elif operation == "convert":
            dtype = params.get("dtype", "float64")
            if dtype not in {"float64", "int64", "string", "category", "datetime64[ns]"}:
                raise FrameworkError("Tipo de conversión no permitido.")
            for c in columns:
                out[c] = (
                    pd.to_datetime(out[c], errors="raise")
                    if dtype == "datetime64[ns]"
                    else out[c].astype(dtype)
                )
        elif operation == "normalize_names":

            def normalize(name: str) -> str:
                plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
                return re.sub("\\W+", "_", plain.strip().lower()).strip("_") or "columna"

            out.columns = [normalize(c) for c in out.columns]
            if not out.columns.is_unique:
                raise FrameworkError("La normalización crea nombres repetidos.")
        elif operation in {"clip", "winsor"}:
            if operation == "winsor":
                q = float(params.get("limit", 0.05))
                if not 0 <= q < 0.5:
                    raise FrameworkError("Límite de winsorización entre 0 y 0.5.")
                low, high = (x.quantile(q), x.quantile(1 - q))
            else:
                q1, q3 = (x.quantile(0.25), x.quantile(0.75))
                low, high = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
            out[columns] = x.clip(low, high, axis=1)
        elif operation == "drop_outliers":
            out = out[
                ~self.outliers(
                    columns, params.get("method", "iqr"), params.get("threshold", 1.5)
                ).any(axis=1)
            ]
        else:
            raise FrameworkError("Transformación desconocida.")
        self._validate(out)
        self._history.append(
            {
                "operacion": operation,
                "columnas": columns,
                "parametros": params,
                "estado": "propuesta",
            }
        )
        return out

    def custom_transform(self, function: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """Provide custom transform using validated inputs."""
        out = function(self.df)
        self._validate(out)
        self._history.append({"operacion": "personalizada", "estado": "propuesta"})
        return out.copy(deep=True)

    def correlation(self, method: str = "pearson") -> pd.DataFrame:
        """Provide correlation using validated inputs."""
        if method not in {"pearson", "spearman", "kendall"}:
            raise FrameworkError("Método de correlación inválido.")
        return self._df.select_dtypes(include=np.number).corr(method=method)

    def vif(self, columns: list[str]) -> pd.Series:
        """Provide vif using validated inputs."""
        from sklearn.linear_model import LinearRegression

        self._columns(columns, True)
        x = self._df[columns].dropna()
        if len(columns) < 2 or len(x) <= len(columns):
            raise FrameworkError("VIF requiere 2+ variables y más filas completas que variables.")
        result = {}
        for c in columns:
            r2 = LinearRegression().fit(x.drop(columns=c), x[c]).score(x.drop(columns=c), x[c])
            result[c] = np.inf if r2 >= 1 - 1e-12 or x[c].nunique() <= 1 else 1 / (1 - r2)
        return pd.Series(result, name="VIF")

    def statistical_test(
        self, name: str, columns: list[str], alpha: float = 0.05
    ) -> dict[str, Any]:
        """Execute a compatible statistical test and report assumption limitations."""
        self._columns(columns, name != "chi2")
        if not 0 < alpha < 1:
            raise FrameworkError("Alpha debe estar entre 0 y 1.")
        warnings = [
            "Supone observaciones independientes; no prueba causalidad.",
            "Al realizar muchas pruebas, ajuste por comparaciones múltiples.",
        ]
        if name == "chi2":
            if len(columns) != 2:
                raise FrameworkError("Chi-cuadrado requiere dos columnas categóricas.")
            table = self.contingency(*columns)
            if min(table.shape) < 2:
                raise FrameworkError("La tabla debe tener al menos 2 filas y 2 columnas.")
            statistic, p, _, expected = stats.chi2_contingency(table)
            if (expected < 5).any():
                raise FrameworkError("Frecuencias esperadas menores de 5; use una prueba exacta.")
        else:
            groups = [self._df[c].dropna().to_numpy(float) for c in columns]
            if any(len(g) < 3 or np.ptp(g) == 0 or (not np.isfinite(g).all()) for g in groups):
                raise FrameworkError("Cada muestra requiere 3+ valores finitos no constantes.")
            if name in {"shapiro", "normaltest"}:
                if len(groups) != 1:
                    raise FrameworkError("La prueba de normalidad requiere una columna.")
                if name == "normaltest" and len(groups[0]) < 8:
                    raise FrameworkError("D'Agostino requiere al menos 8 observaciones.")
                if name == "shapiro" and len(groups[0]) > 5000:
                    groups[0] = np.random.default_rng(self.random_state).choice(
                        groups[0], 5000, replace=False
                    )
                    warnings.append("Shapiro calculado sobre una muestra de 5000 valores.")
                statistic, p = getattr(stats, name)(groups[0])
            else:
                if len(groups) < 2 or (name in {"ttest", "mannwhitney"} and len(groups) != 2):
                    raise FrameworkError("Seleccione un número compatible de muestras.")
                functions = {
                    "levene": stats.levene,
                    "ttest": stats.ttest_ind,
                    "anova": stats.f_oneway,
                    "mannwhitney": stats.mannwhitneyu,
                }
                if name not in functions:
                    raise FrameworkError("Prueba desconocida.")
                if name in {"ttest", "anova"}:
                    normal = all(stats.shapiro(g[:5000]).pvalue > alpha for g in groups)
                    equal = stats.levene(*groups).pvalue > alpha
                    if not normal or not equal:
                        raise FrameworkError(
                            "Normalidad/homogeneidad no compatibles; use Mann-Whitney para dos muestras o revise el diseño."
                        )
                    warnings.append("No rechazar normalidad no garantiza normalidad poblacional.")
                statistic, p = functions[name](*groups)
        return {
            "estadistico": float(statistic),
            "p_value": float(p),
            "alpha": alpha,
            "interpretacion": "Rechazar H0" if p < alpha else "No rechazar H0",
            "advertencias": warnings,
        }

    def chart(
        self,
        kind: str,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        **params: Any,
    ) -> Any:
        """Return a validated figure without importing a frontend framework."""
        from .visualization import plot

        selected = list(dict.fromkeys(c for c in [x, y, color] if c is not None))
        if selected:
            self._columns(selected)
        return plot(self.df, kind, x, y, color, **params)

    def report(self) -> str:
        """Provide report using validated inputs."""
        return html_report(
            "Atlas · EDA",
            {
                "Información": self.information(),
                "Resumen": self.summary(),
                "Nulos": self.missing_values(),
            },
            "Estadísticas exploratorias; asociaciones no implican causalidad.",
        )
