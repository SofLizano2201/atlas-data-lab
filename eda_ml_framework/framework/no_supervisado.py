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
from .visualization import NOISE_COLOR, PALETTE, SYMBOLS, categorical_colors, note, style


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
        if (
            config.get("encoding", "onehot") == "onehot"
            and sum(categories[c].nunique() for c in categories) + len(columnas) > 1000
        ):
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
        result["datos"].attrs["atlas_method"] = metodo
        result["datos"].attrs["atlas_variance"] = (
            result["varianza"].tolist() if result["varianza"] is not None else None
        )
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
        if len(cols) not in {2, 3}:
            raise FrameworkError("La representación visual requiere dos o tres componentes.")
        data = datos.copy()
        method = datos.attrs.get("atlas_method", "")
        variance = datos.attrs.get("atlas_variance")
        axis_labels = {
            column: (
                f"PCA {i + 1} · {variance[i]:.1%} de varianza"
                if method == "pca" and variance is not None
                else f"{method.upper()} {i + 1}"
                if method
                else str(column)
            )
            for i, column in enumerate(cols)
        }
        kwargs = {"color_discrete_sequence": PALETTE, "labels": axis_labels}
        if etiquetas is not None:
            if len(etiquetas) != len(data):
                raise FrameworkError("Las etiquetas no coinciden con las filas representadas.")
            group = "_grupo_visual"
            while group in data:
                group += "_"
            data[group] = ["Ruido" if str(v) == "-1" else f"Grupo {v}" for v in etiquetas]
            colors = categorical_colors(data[group])
            if "Ruido" in colors:
                colors["Ruido"] = NOISE_COLOR
            symbols = (
                ["circle", "diamond", "square", "cross", "x", "circle-open"]
                if len(cols) == 3
                else SYMBOLS
            )
            symbol_map = {name: symbols[i % len(symbols)] for i, name in enumerate(sorted(colors))}
            symbol_map["Ruido"] = "x"
            kwargs = {
                "color_discrete_sequence": PALETTE,
                "color": group,
                "symbol": group,
                "color_discrete_map": colors,
                "symbol_map": symbol_map,
                "labels": {**axis_labels, group: "Grupo"},
            }
        fig = (
            px.scatter_3d(data, x=cols[0], y=cols[1], z=cols[2], **kwargs)
            if len(cols) == 3
            else px.scatter(data, x=cols[0], y=cols[1], **kwargs)
        )
        style(fig, f"Representación {method.upper()} · {len(cols)} dimensiones")
        return note(
            fig, "La proyección simplifica la geometría original. Ruido: símbolo ×, cuando existe."
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
