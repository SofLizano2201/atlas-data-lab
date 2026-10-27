"""Agrupamiento, evaluación y caracterización con tratamiento explícito de ruido."""

import json
from hashlib import sha256
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn import cluster
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

from .exceptions import FrameworkError, OptionalDependencyError
from .metrics import clustering_metrics
from .no_supervisado import NoSupervisado
from .results import Experiment, compare
from .visualization import categorical_colors, style

ALGORITHMS = {
    "kmeans": cluster.KMeans,
    "minibatch": cluster.MiniBatchKMeans,
    "dbscan": cluster.DBSCAN,
    "optics": cluster.OPTICS,
    "agglomerative": cluster.AgglomerativeClustering,
    "gaussian": GaussianMixture,
    "spectral": cluster.SpectralClustering,
    "birch": cluster.Birch,
}


class Clustering(NoSupervisado):
    """Aplicar algoritmos sobre la preparación heredada y explorar sus grupos."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.experimentos: list[Experiment] = []

    @staticmethod
    def algoritmos() -> list[str]:
        """Catálogo incluyendo integraciones opcionales."""
        return [*ALGORITHMS, "hierarchical", "kmedoids"]

    def entrenar(
        self,
        algoritmo: str,
        columnas: list[str],
        scaler: str = "standard",
        encoding: str = "onehot",
        **params: Any,
    ) -> Experiment:
        """Entrenar una vez; conservar la matriz utilizada y etiquetas, incluido ruido."""
        start = perf_counter()
        x = self.preparar(columnas, scaler=scaler, encoding=encoding)
        if algoritmo in {"spectral", "hierarchical", "kmedoids"} and len(x) > 2000:
            raise FrameworkError("Este algoritmo está limitado a 2000 filas.")
        if len(x) < 3:
            raise FrameworkError("Clustering requiere al menos 3 observaciones.")
        count = int(params.get("n_clusters", params.get("n_components", 3)))
        if algoritmo not in {"dbscan", "optics"} and not 2 <= count < len(x):
            raise FrameworkError("Número de grupos debe estar entre 2 y n-1.")
        if algoritmo == "hierarchical":
            tree = linkage(x, method=params.get("method", "ward"))
            labels = fcluster(tree, count, criterion="maxclust") - 1
            model = tree
        else:
            if algoritmo == "kmedoids":
                try:
                    from sklearn_extra.cluster import KMedoids
                except (ImportError, ValueError) as exc:
                    raise OptionalDependencyError(
                        "K-Medoids requiere scikit-learn-extra compatible "
                        "con NumPy; instale '.[medoids]' en un entorno separado si es necesario."
                    ) from exc
                cls = KMedoids
            elif algoritmo in ALGORITHMS:
                cls = ALGORITHMS[algoritmo]
            else:
                raise FrameworkError("Algoritmo desconocido.")
            if "random_state" in cls().get_params():
                params.setdefault("random_state", self.random_state)
            if algoritmo not in {"dbscan", "optics"}:
                params.setdefault(
                    "n_components" if algoritmo == "gaussian" else "n_clusters", count
                )
            model = cls(**params)
            labels = model.fit_predict(x)
        metrics = clustering_metrics(x, labels)
        if hasattr(model, "inertia_"):
            metrics["Inercia"] = float(model.inertia_)
        config = {"columnas": columnas, "scaler": scaler, "encoding": encoding, "params": params}
        cohort = sha256(
            pd.util.hash_pandas_object(self._df, index=True).values.tobytes()
            + json.dumps(config | {"params": {}}, sort_keys=True).encode()
        ).hexdigest()
        result = Experiment(
            algoritmo,
            "clustering",
            metrics,
            perf_counter() - start,
            model,
            self.df.assign(cluster=labels),
            config,
            cohort,
            {"x": x, "labels": labels, "preprocessor": self.preprocessor},
        )
        self.experimentos.append(result)
        result.record_provenance(self._df, len(self.experimentos))
        return result

    def codo(self, columnas: list[str], max_k: int = 10) -> pd.DataFrame:
        """Inercia de KMeans usando una única preparación."""
        x = self.preparar(columnas)
        return pd.DataFrame(
            [
                {
                    "k": k,
                    "inercia": cluster.KMeans(
                        n_clusters=k, random_state=self.random_state, n_init=10
                    )
                    .fit(x)
                    .inertia_,
                }
                for k in range(1, min(max_k, len(x) - 1) + 1)
            ]
        )

    def silhouette_plot(self, resultado: Experiment) -> Any:
        """Silhouette individual excluyendo ruido; máximo 2000 filas."""
        x, labels = resultado.details["x"], resultado.details["labels"]
        keep = labels != -1
        if not 1 < len(np.unique(labels[keep])) < keep.sum() or keep.sum() > 2000:
            raise FrameworkError("Silhouette requiere 2+ grupos válidos y hasta 2000 filas.")
        values = silhouette_samples(x[keep], labels[keep])
        table = pd.DataFrame(
            {"silhouette": values, "cluster": [f"Grupo {v}" for v in labels[keep]]}
        )
        fig = style(
            px.box(
                table,
                x="cluster",
                y="silhouette",
                color="cluster",
                color_discrete_map=categorical_colors(table.cluster),
                labels={"cluster": "Grupo", "silhouette": "Coeficiente de silueta"},
            ),
            "Silueta por grupo · distribución individual, sin ruido",
        )
        fig.update_yaxes(range=[-1, 1])
        fig.add_hline(
            y=float(values.mean()),
            line_dash="dash",
            line_color="#331832",
            annotation_text=f"Media: {values.mean():.3f}",
        )
        return fig

    def dendrograma(self, columnas: list[str], metodo: str = "ward") -> Any:
        """Dendrograma limitado y truncado para legibilidad."""
        if len(self._df) > 2000:
            raise FrameworkError("Dendrograma limitado a 2000 filas.")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(
            linkage(self.preparar(columnas), method=metodo),
            ax=ax,
            truncate_mode="lastp",
            p=30,
            link_color_func=lambda _: "#1B5299",
        )
        ax.set(
            title=f"Dendrograma · enlace {metodo} · hasta 30 ramas",
            xlabel="Observación o grupo agregado",
            ylabel="Distancia de enlace",
        )
        fig.tight_layout()
        return fig

    def caracterizar(self, resultado: Experiment) -> pd.DataFrame:
        """Estadísticas por cluster, incluyendo grupo -1 de ruido."""
        columns = list(self._df.select_dtypes(include=np.number).columns)
        return resultado.predictions.groupby("cluster")[columns].agg(["count", "mean", "std"])

    def representantes(self, resultado: Experiment) -> pd.DataFrame:
        """Media en espacio transformado de cada grupo, excluyendo ruido."""
        x = pd.DataFrame(resultado.details["x"])
        x["cluster"] = resultado.details["labels"]
        return x[x.cluster != -1].groupby("cluster").mean()

    def comparar(self) -> pd.DataFrame:
        """Comparar resultados compatibles almacenados."""
        return compare(self.experimentos)
