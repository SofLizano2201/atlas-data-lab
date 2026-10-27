"""Clase madre: exploración, diagnóstico y transformaciones con versiones."""

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

from contracts import DatasetVersion

from .exceptions import FrameworkError
from .preprocessing import SCALERS
from .provenance import fingerprint
from .results import export_frame, html_report


class AnalisisDatosExploratorio:
    """Conservar datos encapsulados; propuestas no mutan hasta confirmar aplicar()."""

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
        if len(df) > 100_000 or len(df.columns) > 500:
            raise FrameworkError("Límite interactivo: 100 000 filas y 500 columnas.")
        if df.memory_usage(deep=True).sum() > 100 * 1024**2:
            raise FrameworkError("El DataFrame supera 100 MB en memoria.")

    @property
    def original(self) -> pd.DataFrame:
        """Original encapsulado: cada acceso entrega una copia defensiva."""
        return self._original.copy(deep=True)

    @property
    def dataset_version(self) -> DatasetVersion:
        """Identidad del original y huella de la versión actual."""
        applied = [h for h in self._history if h.get("estado") == "aplicada"]
        return DatasetVersion(
            dataset_id=self._dataset_id,
            version=len(self._versions) - 1,
            content_hash=self._content_hash,
            rows=len(self._df),
            columns=self.columnas(),
            operation=str(applied[-1]["operacion"]) if applied else "carga",
        )

    @property
    def df(self) -> pd.DataFrame:
        """Copia defensiva del dataset de trabajo."""
        return self._df.copy(deep=True)

    @property
    def historial(self) -> pd.DataFrame:
        """Operaciones propuestas, aplicadas y restauradas."""
        return pd.DataFrame(self._history)

    @property
    def versiones(self) -> int:
        """Cantidad de instantáneas disponibles."""
        return len(self._versions)

    @classmethod
    def cargar(
        cls,
        source: Any,
        formato: str | None = None,
        separador: str = ",",
        codificacion: str = "utf-8",
        hoja: str | int = 0,
    ) -> AnalisisDatosExploratorio:
        """Leer formatos tabulares sin ejecutar código; limitar bytes y tamaño descomprimido."""
        name = getattr(source, "name", str(source))
        fmt = (formato or Path(name).suffix.lstrip(".")).lower()
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
                if len(separador) != 1:
                    raise FrameworkError("El separador debe ser un carácter.")
                frame = pd.read_csv(buffer, sep=separador, encoding=codificacion, nrows=100_001)
            elif fmt == "xlsx":
                from zipfile import ZipFile

                with ZipFile(buffer) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 100 * 1024**2:
                        raise FrameworkError("Excel descomprimido supera 100 MB.")
                buffer.seek(0)
                frame = pd.read_excel(buffer, sheet_name=hoja, nrows=100_001)
            elif fmt == "json":
                frame = pd.read_json(buffer)
            else:
                import pyarrow.parquet as pq

                metadata = pq.ParquetFile(buffer).metadata
                if (
                    metadata.num_rows > 100_000
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
    def cargar_csv(cls, source: Any, **config: Any) -> AnalisisDatosExploratorio:
        """Cargar CSV con separador y codificación configurables."""
        return cls.cargar(source, formato="csv", **config)

    @classmethod
    def cargar_excel(cls, source: Any, hoja: str | int = 0) -> AnalisisDatosExploratorio:
        """Cargar una hoja XLSX."""
        return cls.cargar(source, formato="xlsx", hoja=hoja)

    @classmethod
    def cargar_json(cls, source: Any) -> AnalisisDatosExploratorio:
        """Cargar JSON tabular."""
        return cls.cargar(source, formato="json")

    @classmethod
    def cargar_parquet(cls, source: Any) -> AnalisisDatosExploratorio:
        """Cargar Parquet con validación de metadatos."""
        return cls.cargar(source, formato="parquet")

    def vista_previa(self, n: int = 10) -> pd.DataFrame:
        """Primeras filas."""
        return self.df.head(n)

    def dimensiones(self) -> tuple[int, int]:
        """Número de filas y columnas."""
        return self._df.shape

    def columnas(self) -> list[str]:
        """Nombres de columnas."""
        return list(self._df.columns)

    def tipos_datos(self) -> pd.Series:
        """Tipos pandas por columna."""
        return self._df.dtypes.astype(str)

    def memoria(self) -> pd.Series:
        """Consumo profundo en bytes."""
        return self._df.memory_usage(deep=True)

    def informacion(self) -> pd.DataFrame:
        """Diagnóstico general por variable."""
        return pd.DataFrame(
            {
                "tipo": self.tipos_datos(),
                "no_nulos": self._df.count(),
                "unicos": self._df.nunique(),
                "faltantes_%": self.nulos().porcentaje,
                "bytes": self.memoria().drop("Index", errors="ignore"),
            }
        )

    def valores_unicos(self, columna: str) -> np.ndarray:
        """Valores distintos, incluyendo faltantes."""
        self._columns([columna])
        return self._df[columna].unique()

    def constantes(self) -> list[str]:
        """Columnas con un único valor, incluyendo NaN."""
        return list(self._df.columns[self._df.nunique(dropna=False) <= 1])

    def duplicados(self) -> pd.DataFrame:
        """Todas las filas que participan en duplicación."""
        return self.df[self._df.duplicated(keep=False)]

    def copia_seguridad(self) -> int:
        """Guardar una versión; devolver su índice."""
        if len(self._versions) >= 20:
            raise FrameworkError("Máximo de 20 versiones; exporte y cree una nueva sesión.")
        self._validate_version_memory(self._df)
        self._versions.append(self.df)
        return len(self._versions) - 1

    def _validate_version_memory(self, frame: pd.DataFrame) -> None:
        """Limitar las copias antes de modificar datos para conservar atomicidad."""
        used = sum(v.memory_usage(deep=True).sum() for v in self._versions)
        if used + frame.memory_usage(deep=True).sum() > 256 * 1024**2:
            raise FrameworkError(
                "Historial de versiones limitado a 256 MB; exporte antes de continuar."
            )

    def aplicar(self, nuevo: pd.DataFrame, operacion: str, confirmar: bool = False) -> None:
        """Modificar datos exclusivamente con confirmación explícita."""
        if not confirmar:
            raise FrameworkError("Confirme explícitamente la modificación.")
        self._validate(nuevo)
        if len(self._versions) >= 20:
            raise FrameworkError("Límite de versiones alcanzado.")
        self._validate_version_memory(nuevo)
        new_hash = fingerprint(nuevo)
        self._df = nuevo.copy(deep=True)
        self._content_hash = new_hash
        version = self.copia_seguridad()
        self._history.append({"operacion": operacion, "estado": "aplicada", "version": version})

    def restaurar(self, version: int, confirmar: bool = False) -> None:
        """Restaurar conservando versiones anteriores."""
        if not 0 <= version < len(self._versions):
            raise FrameworkError("Versión inexistente.")
        self.aplicar(self._versions[version], f"restaurar {version}", confirmar)

    def exportar(self, formato: str = "csv") -> bytes:
        """Exportar el estado actual."""
        return export_frame(self._df, formato)

    def _columns(self, columnas: list[str], numericas: bool = False) -> None:
        if (
            not columnas
            or len(set(columnas)) != len(columnas)
            or any(c not in self._df for c in columnas)
        ):
            raise FrameworkError("Seleccione columnas existentes y sin repeticiones.")
        if numericas and any(not pd.api.types.is_numeric_dtype(self._df[c]) for c in columnas):
            raise FrameworkError("La operación requiere columnas numéricas.")

    def estadisticas(self, columnas: list[str] | None = None) -> pd.DataFrame:
        """Media, dispersión, forma y cuartiles; faltantes excluidos."""
        columnas = (
            columnas
            if columnas is not None
            else list(self._df.select_dtypes(include=np.number).columns)
        )
        self._columns(columnas, True)
        x = self._df[columnas]
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

    def estadistica(self, nombre: str, columna: str) -> float:
        """Consultar una estadística individual del catálogo numérico."""
        table = self.estadisticas([columna])
        if nombre not in table:
            raise FrameworkError("Estadística desconocida.")
        return float(table.loc[columna, nombre])

    def media(self, columna: str) -> float:
        """Media aritmética excluyendo nulos."""
        return self.estadistica("media", columna)

    def mediana(self, columna: str) -> float:
        """Mediana excluyendo nulos."""
        return self.estadistica("mediana", columna)

    def moda(self, columna: str) -> pd.Series:
        """Todas las modas, también para variables categóricas."""
        self._columns([columna])
        return self._df[columna].mode()

    def minimo(self, columna: str) -> float:
        """Mínimo numérico."""
        return self.estadistica("minimo", columna)

    def maximo(self, columna: str) -> float:
        """Máximo numérico."""
        return self.estadistica("maximo", columna)

    def rango(self, columna: str) -> float:
        """Máximo menos mínimo."""
        return self.estadistica("rango", columna)

    def varianza(self, columna: str) -> float:
        """Varianza muestral, ddof=1."""
        return self.estadistica("varianza", columna)

    def desviacion_estandar(self, columna: str) -> float:
        """Desviación estándar muestral."""
        return self.estadistica("desviacion", columna)

    def cuartiles(self, columna: str) -> pd.Series:
        """Cuartiles primero, segundo y tercero."""
        return self.percentiles(columna, [25, 50, 75])

    def rango_intercuartilico(self, columna: str) -> float:
        """Tercer cuartil menos primer cuartil."""
        return self.estadistica("iqr", columna)

    def coeficiente_variacion(self, columna: str) -> float:
        """Desviación / valor absoluto de la media; indefinido con media cero."""
        return self.estadistica("cv", columna)

    def asimetria(self, columna: str) -> float:
        """Asimetría muestral ajustada."""
        return self.estadistica("asimetria", columna)

    def curtosis(self, columna: str) -> float:
        """Exceso de curtosis muestral."""
        return self.estadistica("curtosis", columna)

    def percentiles(self, columna: str, valores: list[float]) -> pd.Series:
        """Percentiles expresados entre 0 y 100."""
        self._columns([columna], True)
        if any(not 0 <= v <= 100 for v in valores):
            raise FrameworkError("Percentiles fuera de [0,100].")
        return self._df[columna].quantile(np.asarray(valores) / 100)

    def frecuencias(self, columna: str) -> pd.DataFrame:
        """Frecuencias absolutas y relativas incluyendo faltantes."""
        self._columns([columna])
        n = self._df[columna].value_counts(dropna=False)
        return pd.DataFrame({"absoluta": n, "relativa": n / len(self._df)})

    def contingencia(self, a: str, b: str) -> pd.DataFrame:
        """Tabla de contingencia de dos categorías."""
        self._columns([a, b])
        if (self._df[a].nunique(dropna=False) * self._df[b].nunique(dropna=False)) > 10000:
            raise FrameworkError("Contingencia limitada a 10000 celdas; agrupe categorías primero.")
        return pd.crosstab(self._df[a].fillna("(faltante)"), self._df[b].fillna("(faltante)"))

    def resumen(self) -> pd.DataFrame:
        """Resumen de todas las variables."""
        return self._df.describe(include="all").T

    def estadisticas_agrupadas(self, categoria: str, columna: str) -> pd.DataFrame:
        """Resumen numérico por categoría."""
        self._columns([categoria, columna])
        self._columns([columna], True)
        return self._df.groupby(categoria, dropna=False, observed=True)[columna].describe()

    def nulos(self) -> pd.DataFrame:
        """Número y porcentaje de faltantes."""
        n = self._df.isna().sum()
        return pd.DataFrame({"cantidad": n, "porcentaje": 100 * n / len(self._df)})

    def outliers(
        self, columnas: list[str], metodo: str = "iqr", umbral: float = 1.5
    ) -> pd.DataFrame:
        """Máscara por celda; Isolation Forest produce una máscara por fila."""
        self._columns(columnas, True)
        if umbral <= 0:
            raise FrameworkError("El umbral debe ser positivo.")
        x = self._df[columnas]
        if metodo == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            return (x < q1 - umbral * (q3 - q1)) | (x > q3 + umbral * (q3 - q1))
        if metodo == "zscore":
            return ((x - x.mean()) / x.std(ddof=0).replace(0, np.nan)).abs() > umbral
        if metodo == "isolation":
            if x.isna().all().any():
                raise FrameworkError("Impute columnas completamente vacías primero.")
            mask = (
                IsolationForest(random_state=self.random_state).fit_predict(x.fillna(x.median()))
                == -1
            )
            return pd.DataFrame({c: mask for c in columnas}, index=x.index)
        raise FrameworkError("Método de outliers desconocido.")

    def inconsistencias(self) -> pd.DataFrame:
        """Diagnosticar textos convertibles a número y mezclas de tipos."""
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

    def categorias_raras(self, columna: str, umbral: float = 0.01) -> pd.DataFrame:
        """Categorías con frecuencia relativa menor al umbral."""
        if not 0 <= umbral <= 1:
            raise FrameworkError("Umbral entre 0 y 1.")
        table = self.frecuencias(columna)
        return table[table.relativa < umbral]

    def transformar(
        self, operacion: str, columnas: list[str] | None = None, **params: Any
    ) -> pd.DataFrame:
        """Proponer un nuevo DataFrame y registrar parámetros, sin modificar el original."""
        columns = columnas if columnas is not None else self.columnas()
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
        if operacion in numeric_ops:
            self._columns(columns, True)
        if operacion == "drop_rows":
            out = out.dropna(subset=columns)
        elif operacion == "drop_null_columns":
            threshold = float(params.get("umbral", 0.5))
            if not 0 <= threshold <= 1:
                raise FrameworkError("Umbral entre 0 y 1.")
            out = out.drop(columns=[c for c in columns if out[c].isna().mean() > threshold])
        elif operacion == "duplicates":
            out = out.drop_duplicates(subset=columns)
        elif operacion in {"mean", "median", "mode", "constant"}:
            for c in columns:
                value = (
                    x[c].mean()
                    if operacion == "mean"
                    else x[c].median()
                    if operacion == "median"
                    else params.get("valor", 0)
                    if operacion == "constant"
                    else x[c].mode().iloc[0]
                    if not x[c].mode().empty
                    else np.nan
                )
                if pd.isna(value):
                    raise FrameworkError(f"{c}: no hay valores para imputar.")
                out[c] = out[c].fillna(value)
        elif operacion == "knn":
            if len(x) > 5000:
                raise FrameworkError("Imputación KNN interactiva limitada a 5000 filas.")
            out[columns] = KNNImputer(
                n_neighbors=int(params.get("n_neighbors", 5)), keep_empty_features=True
            ).fit_transform(x)
        elif operacion in SCALERS:
            out[columns] = SCALERS[operacion]().fit_transform(x)
        elif operacion in {"onehot", "ordinal"}:
            if operacion == "onehot" and (
                sum(x[c].nunique(dropna=False) for c in columns) * len(x) > 10_000_000
            ):
                raise FrameworkError(
                    "One-hot excedería 10 millones de celdas; excluya identificadores."
                )
            encoder = (
                OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                if (operacion == "onehot")
                else OrdinalEncoder()
            )
            encoded = encoder.fit_transform(x.astype(str))
            names = encoder.get_feature_names_out(columns)
            if set(names) & (set(out.columns) - set(columns)):
                raise FrameworkError("La codificación produciría nombres duplicados.")
            out = pd.concat(
                [out.drop(columns=columns), pd.DataFrame(encoded, columns=names, index=out.index)],
                axis=1,
            )
        elif operacion == "label":
            if len(columns) != 1 or x.isna().any().any():
                raise FrameworkError("LabelEncoder requiere una sola variable objetivo sin nulos.")
            out[columns[0]] = LabelEncoder().fit_transform(x.iloc[:, 0])
        elif operacion == "log":
            if (x <= 0).any().any():
                raise FrameworkError("Log requiere valores estrictamente positivos.")
            out[columns] = np.log(x)
        elif operacion in {"boxcox", "yeojohnson"}:
            out[columns] = PowerTransformer(
                method="box-cox" if operacion == "boxcox" else "yeo-johnson"
            ).fit_transform(x)
        elif operacion == "binning":
            out[columns] = KBinsDiscretizer(
                n_bins=int(params.get("n_bins", 5)),
                encode="ordinal",
                strategy="quantile",
                subsample=None,
                quantile_method="averaged_inverted_cdf",
            ).fit_transform(x)
        elif operacion == "select":
            out = out[columns]
        elif operacion == "drop_columns":
            out = out.drop(columns=columns)
        elif operacion == "convert":
            dtype = params.get("tipo", "float64")
            if dtype not in {"float64", "int64", "string", "category", "datetime64[ns]"}:
                raise FrameworkError("Tipo de conversión no permitido.")
            for c in columns:
                out[c] = (
                    pd.to_datetime(out[c], errors="raise")
                    if dtype == "datetime64[ns]"
                    else (out[c].astype(dtype))
                )
        elif operacion == "normalize_names":

            def normalize(name: str) -> str:
                plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
                return re.sub(r"\W+", "_", plain.strip().lower()).strip("_") or "columna"

            out.columns = [normalize(c) for c in out.columns]
            if not out.columns.is_unique:
                raise FrameworkError("La normalización crea nombres repetidos.")
        elif operacion in {"clip", "winsor"}:
            if operacion == "winsor":
                q = float(params.get("limite", 0.05))
                if not 0 <= q < 0.5:
                    raise FrameworkError("Límite de winsorización entre 0 y 0.5.")
                low, high = x.quantile(q), x.quantile(1 - q)
            else:
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            out[columns] = x.clip(low, high, axis=1)
        elif operacion == "drop_outliers":
            out = out[
                ~self.outliers(columns, params.get("metodo", "iqr"), params.get("umbral", 1.5)).any(
                    axis=1
                )
            ]
        else:
            raise FrameworkError("Transformación desconocida.")
        self._validate(out)
        self._history.append(
            {
                "operacion": operacion,
                "columnas": columns,
                "parametros": params,
                "estado": "propuesta",
            }
        )
        return out

    def transformar_personalizado(
        self, funcion: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """API Python para funciones de confianza; el frontend no acepta código."""
        out = funcion(self.df)
        self._validate(out)
        self._history.append({"operacion": "personalizada", "estado": "propuesta"})
        return out.copy(deep=True)

    def correlacion(self, metodo: str = "pearson") -> pd.DataFrame:
        """Correlación por pares con valores faltantes excluidos."""
        if metodo not in {"pearson", "spearman", "kendall"}:
            raise FrameworkError("Método de correlación inválido.")
        return self._df.select_dtypes(include=np.number).corr(method=metodo)

    def vif(self, columnas: list[str]) -> pd.Series:
        """VIF por regresión auxiliar con intercepto; colinealidad perfecta da infinito."""
        from sklearn.linear_model import LinearRegression

        self._columns(columnas, True)
        x = self._df[columnas].dropna()
        if len(columnas) < 2 or len(x) <= len(columnas):
            raise FrameworkError("VIF requiere 2+ variables y más filas completas que variables.")
        result = {}
        for c in columnas:
            r2 = LinearRegression().fit(x.drop(columns=c), x[c]).score(x.drop(columns=c), x[c])
            result[c] = np.inf if r2 >= 1 - 1e-12 or x[c].nunique() <= 1 else 1 / (1 - r2)
        return pd.Series(result, name="VIF")

    def prueba(self, nombre: str, columnas: list[str], alpha: float = 0.05) -> dict[str, Any]:
        """Pruebas exploratorias; observaciones independientes deben justificarse externamente."""
        self._columns(columnas, nombre != "chi2")
        if not 0 < alpha < 1:
            raise FrameworkError("Alpha debe estar entre 0 y 1.")
        warnings = [
            "Supone observaciones independientes; no prueba causalidad.",
            "Al realizar muchas pruebas, ajuste por comparaciones múltiples.",
        ]
        if nombre == "chi2":
            if len(columnas) != 2:
                raise FrameworkError("Chi-cuadrado requiere dos columnas categóricas.")
            table = self.contingencia(*columnas)
            if min(table.shape) < 2:
                raise FrameworkError("La tabla debe tener al menos 2 filas y 2 columnas.")
            statistic, p, _, expected = stats.chi2_contingency(table)
            if (expected < 5).any():
                raise FrameworkError("Frecuencias esperadas menores de 5; use una prueba exacta.")
        else:
            groups = [self._df[c].dropna().to_numpy(float) for c in columnas]
            if any(len(g) < 3 or np.ptp(g) == 0 or not np.isfinite(g).all() for g in groups):
                raise FrameworkError("Cada muestra requiere 3+ valores finitos no constantes.")
            if nombre in {"shapiro", "normaltest"}:
                if len(groups) != 1:
                    raise FrameworkError("La prueba de normalidad requiere una columna.")
                if nombre == "normaltest" and len(groups[0]) < 8:
                    raise FrameworkError("D'Agostino requiere al menos 8 observaciones.")
                if nombre == "shapiro" and len(groups[0]) > 5000:
                    groups[0] = np.random.default_rng(self.random_state).choice(
                        groups[0], 5000, replace=False
                    )
                    warnings.append("Shapiro calculado sobre una muestra de 5000 valores.")
                statistic, p = getattr(stats, nombre)(groups[0])
            else:
                if len(groups) < 2 or (nombre in {"ttest", "mannwhitney"} and len(groups) != 2):
                    raise FrameworkError("Seleccione un número compatible de muestras.")
                functions = {
                    "levene": stats.levene,
                    "ttest": stats.ttest_ind,
                    "anova": stats.f_oneway,
                    "mannwhitney": stats.mannwhitneyu,
                }
                if nombre not in functions:
                    raise FrameworkError("Prueba desconocida.")
                if nombre in {"ttest", "anova"}:
                    normal = all(stats.shapiro(g[:5000]).pvalue > alpha for g in groups)
                    equal = stats.levene(*groups).pvalue > alpha
                    if not normal or not equal:
                        raise FrameworkError(
                            "Normalidad/homogeneidad no compatibles; use Mann-Whitney "
                            "para dos muestras o revise el diseño."
                        )
                    warnings.append("No rechazar normalidad no garantiza normalidad poblacional.")
                statistic, p = functions[nombre](*groups)
        return {
            "estadistico": float(statistic),
            "p_value": float(p),
            "alpha": alpha,
            "interpretacion": "Rechazar H0" if p < alpha else "No rechazar H0",
            "advertencias": warnings,
        }

    def grafico(
        self,
        tipo: str,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        **params: Any,
    ) -> Any:
        """Devolver figura sin ninguna dependencia de Streamlit."""
        from .visualization import plot

        selected = list(dict.fromkeys(c for c in [x, y, color] if c is not None))
        if selected:
            self._columns(selected)
        return plot(self.df, tipo, x, y, color, **params)

    def reporte(self) -> str:
        """Reporte HTML de exploración."""
        return html_report(
            "Atlas · EDA",
            {"Información": self.informacion(), "Resumen": self.resumen(), "Nulos": self.nulos()},
            "Estadísticas exploratorias; asociaciones no implican causalidad.",
        )
