"""Herramientas compartidas de representación y preprocesamiento."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .eda import AnalisisDatosExploratorio
from .exceptions import FrameworkError, OptionalDependencyError
from .preprocessing import make_preprocessor, validate_features
from .visualization import PALETTE, style


class NoSupervisado(AnalisisDatosExploratorio):
    """Hereda EDA y añade representaciones sin variable objetivo."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.preprocessor: Any = None
        self.representations: dict[str, dict[str, Any]] = {}

    def preparar(self, columnas: list[str], **config: Any) -> np.ndarray:
        """Ajustar preprocesamiento exploratorio sobre variables seleccionadas."""
        self._columns(columnas)
        if len(self._df) > 5000:
            raise FrameworkError(
                "Representación interactiva limitada a 5000 filas; seleccione muestra."
            )
        x = validate_features(self._df[columnas])
        categories = x.select_dtypes(exclude=np.number)
        if sum(categories[c].nunique() for c in categories) + len(columnas) > 1000:
            raise FrameworkError("Exceso de categorías: excluya identificadores o use ordinal.")
        self.preprocessor = make_preprocessor(x, **config)
        return self.preprocessor.fit_transform(x)

    def reducir(
        self,
        columnas: list[str],
        metodo: str = "pca",
        dimensiones: int = 2,
        scaler: str = "standard",
        encoding: str = "onehot",
        **params: Any,
    ) -> dict[str, Any]:
        """PCA, t-SNE o UMAP; las dos últimas son representaciones exploratorias."""
        x = self.preparar(columnas, scaler=scaler, encoding=encoding)
        if dimensiones not in {2, 3} or dimensiones > min(x.shape):
            raise FrameworkError("Use 2 o 3 dimensiones, sin exceder filas ni características.")
        if metodo == "pca":
            model = PCA(n_components=dimensiones, random_state=self.random_state, **params)
        elif metodo == "tsne":
            params.setdefault("perplexity", min(30, max(1, (len(x) - 1) / 3)))
            if params["perplexity"] >= len(x):
                raise FrameworkError("Perplexity debe ser menor que el número de filas.")
            model = TSNE(n_components=dimensiones, random_state=self.random_state, **params)
        elif metodo == "umap":
            try:
                from umap import UMAP
            except (ImportError, ValueError) as exc:
                raise OptionalDependencyError(
                    "Instale la integración: pip install '.[umap]'"
                ) from exc
            model = UMAP(n_components=dimensiones, random_state=self.random_state, **params)
        else:
            raise FrameworkError("Reducción desconocida.")
        embedding = model.fit_transform(x)
        result = {
            "datos": pd.DataFrame(
                embedding,
                index=self._df.index,
                columns=[f"Componente {i + 1}" for i in range(dimensiones)],
            ),
            "modelo": model,
            "varianza": getattr(model, "explained_variance_ratio_", None),
            "advertencia": "t-SNE y UMAP no preservan toda la geometría original.",
        }
        self.representations[metodo] = result
        return result

    def distancias(self, columnas: list[str], metrica: str = "euclidean") -> pd.DataFrame:
        """Matriz de distancias limitada a 1000 observaciones."""
        if metrica not in {"euclidean", "cityblock", "cosine"} or len(self._df) > 1000:
            raise FrameworkError("Use euclidean, cityblock o cosine con hasta 1000 filas.")
        x = self.preparar(columnas)
        return pd.DataFrame(
            cdist(x, x, metric=metrica), index=self._df.index, columns=self._df.index
        )

    def visualizar_representacion(self, datos: pd.DataFrame, etiquetas: Any = None) -> Any:
        """Figura 2D o 3D con colores categóricos."""
        cols = list(datos.columns)
        color = None if etiquetas is None else np.asarray(etiquetas).astype(str)
        if len(cols) == 3:
            return style(
                px.scatter_3d(
                    datos,
                    x=cols[0],
                    y=cols[1],
                    z=cols[2],
                    color=color,
                    color_discrete_sequence=PALETTE,
                )
            )
        return style(
            px.scatter(datos, x=cols[0], y=cols[1], color=color, color_discrete_sequence=PALETTE)
        )

    def comparar_representaciones(self) -> pd.DataFrame:
        """Resumen de las representaciones ya calculadas, sin repetir ajustes."""
        return pd.DataFrame(
            [
                {
                    "metodo": k,
                    "dimensiones": v["datos"].shape[1],
                    "varianza_acumulada": np.sum(v["varianza"])
                    if v["varianza"] is not None
                    else np.nan,
                }
                for k, v in self.representations.items()
            ]
        )
