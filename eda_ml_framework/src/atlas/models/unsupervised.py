"""Scientific model implementation, independent of agents and Streamlit."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from atlas.core.exceptions import FrameworkError, OptionalDependencyError

from .eda import ExploratoryDataAnalysis
from .preprocessing import make_preprocessor, validate_features
from .visualization import NOISE_COLOR, PALETTE, SYMBOLS, categorical_colors, note, style


class UnsupervisedLearning(ExploratoryDataAnalysis):
    """Extend exploratory analysis with preprocessing, distances and representations."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.preprocessor: Any = None
        self.representations: dict[str, dict[str, Any]] = {}

    def prepare(self, columns: list[str], **config: Any) -> np.ndarray:
        """Provide prepare using validated inputs."""
        self._columns(columns)
        if len(self._df) > 5000:
            raise FrameworkError(
                "Representación interactiva limitada a 5000 filas; seleccione muestra."
            )
        x = validate_features(self._df[columns])
        categories = x.select_dtypes(exclude=np.number)
        if (
            config.get("encoding", "onehot") == "onehot"
            and sum(categories[c].nunique() for c in categories) + len(columns) > 1000
        ):
            raise FrameworkError("Exceso de categorías: excluya identificadores o use ordinal.")
        self.preprocessor = make_preprocessor(x, **config)
        return self.preprocessor.fit_transform(x)

    def reduce_dimensions(
        self,
        columns: list[str],
        method: str = "pca",
        dimensions: int = 2,
        scaler: str = "standard",
        encoding: str = "onehot",
        **params: Any,
    ) -> dict[str, Any]:
        """Fit PCA, t-SNE or optional UMAP for exploratory representation."""
        x = self.prepare(columns, scaler=scaler, encoding=encoding)
        if dimensions not in {2, 3} or dimensions > min(x.shape):
            raise FrameworkError("Use 2 o 3 dimensiones, sin exceder filas ni características.")
        if method == "pca":
            model = PCA(n_components=dimensions, random_state=self.random_state, **params)
        elif method == "tsne":
            params.setdefault("perplexity", min(30, max(1, (len(x) - 1) / 3)))
            if params["perplexity"] >= len(x):
                raise FrameworkError("Perplexity debe ser menor que el número de filas.")
            model = TSNE(n_components=dimensions, random_state=self.random_state, **params)
        elif method == "umap":
            try:
                from umap import UMAP
            except (ImportError, ValueError) as exc:
                raise OptionalDependencyError(
                    "Instale la integración: pip install '.[umap]'"
                ) from exc
            model = UMAP(n_components=dimensions, random_state=self.random_state, **params)
        else:
            raise FrameworkError("Reducción desconocida.")
        embedding = model.fit_transform(x)
        result = {
            "datos": pd.DataFrame(
                embedding,
                index=self._df.index,
                columns=[f"Componente {i + 1}" for i in range(dimensions)],
            ),
            "modelo": model,
            "varianza": getattr(model, "explained_variance_ratio_", None),
            "advertencia": "t-SNE y UMAP no preservan toda la geometría original.",
        }
        result["datos"].attrs["atlas_method"] = method
        result["explained_variance_total"] = (
            float(np.sum(result["varianza"])) if result["varianza"] is not None else None
        )
        result["datos"].attrs["atlas_variance"] = (
            result["varianza"].tolist() if result["varianza"] is not None else None
        )
        self.representations[method] = result
        return result

    def distances(self, columns: list[str], metric: str = "euclidean") -> pd.DataFrame:
        """Provide distances using validated inputs."""
        if metric not in {"euclidean", "cityblock", "cosine"} or len(self._df) > 1000:
            raise FrameworkError("Use euclidean, cityblock o cosine con hasta 1000 filas.")
        x = self.prepare(columns)
        return pd.DataFrame(
            cdist(x, x, metric=metric), index=self._df.index, columns=self._df.index
        )

    def plot_representation(self, data: pd.DataFrame, labels: Any = None) -> Any:
        """Provide plot representation using validated inputs."""
        cols = list(data.columns)
        if len(cols) not in {2, 3}:
            raise FrameworkError("La representación visual requiere dos o tres componentes.")
        data = data.copy()
        method = data.attrs.get("atlas_method", "")
        variance = data.attrs.get("atlas_variance")
        axis_labels = {
            column: f"PCA {i + 1} · {variance[i]:.1%} de varianza"
            if method == "pca" and variance is not None
            else f"{method.upper()} {i + 1}"
            if method
            else str(column)
            for i, column in enumerate(cols)
        }
        kwargs = {"color_discrete_sequence": PALETTE, "labels": axis_labels}
        if labels is not None:
            if len(labels) != len(data):
                raise FrameworkError("Las etiquetas no coinciden con las filas representadas.")
            group = "_grupo_visual"
            while group in data:
                group += "_"
            data[group] = ["Ruido" if str(v) == "-1" else f"Grupo {v}" for v in labels]
            colors = categorical_colors(data[group], max_categories=24)
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

    def compare_representations(self) -> pd.DataFrame:
        """Provide compare representations using validated inputs."""
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
