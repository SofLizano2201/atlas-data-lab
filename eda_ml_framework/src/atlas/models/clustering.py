"""Scientific model implementation, independent of agents and Streamlit."""

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

from atlas.core.exceptions import FrameworkError, OptionalDependencyError

from .experiment import Experiment, compare
from .metrics import clustering_metrics
from .unsupervised import UnsupervisedLearning
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


class Clustering(UnsupervisedLearning):
    """Fit, evaluate and characterize clusters while treating noise explicitly."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42) -> None:
        super().__init__(df, random_state)
        self.experiments: list[Experiment] = []

    @staticmethod
    def algorithms() -> list[str]:
        """Provide algorithms using validated inputs."""
        return [*ALGORITHMS, "hierarchical", "kmedoids"]

    def train(
        self,
        algorithm: str,
        columns: list[str],
        scaler: str = "standard",
        encoding: str = "onehot",
        **params: Any,
    ) -> Experiment:
        """Fit the existing estimator workflow and record reproducible experiment results."""
        start = perf_counter()
        x = self.prepare(columns, scaler=scaler, encoding=encoding)
        if algorithm in {"spectral", "hierarchical", "kmedoids"} and len(x) > 2000:
            raise FrameworkError("Este algoritmo está limitado a 2000 filas.")
        if len(x) < 3:
            raise FrameworkError("Clustering requiere al menos 3 observaciones.")
        count = int(params.get("n_clusters", params.get("n_components", 3)))
        if algorithm not in {"dbscan", "optics"} and (not 2 <= count < len(x)):
            raise FrameworkError("Número de grupos debe estar entre 2 y n-1.")
        if algorithm == "hierarchical":
            tree = linkage(x, method=params.get("method", "ward"))
            labels = fcluster(tree, count, criterion="maxclust") - 1
            model = tree
        else:
            if algorithm == "kmedoids":
                try:
                    from sklearn_extra.cluster import KMedoids
                except (ImportError, ValueError) as exc:
                    raise OptionalDependencyError(
                        "K-Medoids requiere scikit-learn-extra compatible con NumPy; instale '.[medoids]' en un entorno separado si es necesario."
                    ) from exc
                cls = KMedoids
            elif algorithm in ALGORITHMS:
                cls = ALGORITHMS[algorithm]
            else:
                raise FrameworkError("Algoritmo desconocido.")
            if "random_state" in cls().get_params():
                params.setdefault("random_state", self.random_state)
            if algorithm not in {"dbscan", "optics"}:
                params.setdefault(
                    "n_components" if algorithm == "gaussian" else "n_clusters", count
                )
            model = cls(**params)
            labels = model.fit_predict(x)
        metrics = clustering_metrics(x, labels)
        if hasattr(model, "inertia_"):
            metrics["Inercia"] = float(model.inertia_)
        config = {"columns": columns, "scaler": scaler, "encoding": encoding, "params": params}
        cohort = sha256(
            pd.util.hash_pandas_object(self._df, index=True).values.tobytes()
            + json.dumps(config | {"params": {}}, sort_keys=True).encode()
        ).hexdigest()
        result = Experiment(
            algorithm,
            "clustering",
            metrics,
            perf_counter() - start,
            model,
            self.df.assign(cluster=labels),
            config,
            cohort,
            {"x": x, "labels": labels, "preprocessor": self.preprocessor},
        )
        self.experiments.append(result)
        result.record_provenance(self._df, len(self.experiments))
        return result

    def elbow(self, columns: list[str], max_k: int = 10, **preprocessing: Any) -> pd.DataFrame:
        """Provide elbow using validated inputs."""
        x = self.prepare(columns, **preprocessing)
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

    def silhouette_plot(self, result: Experiment) -> Any:
        """Provide silhouette plot using validated inputs."""
        x, labels = (result.details["x"], result.details["labels"])
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

    def dendrogram(self, columns: list[str], method: str = "ward", **preprocessing: Any) -> Any:
        """Provide dendrogram using validated inputs."""
        if len(self._df) > 2000:
            raise FrameworkError("Dendrograma limitado a 2000 filas.")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(
            linkage(self.prepare(columns, **preprocessing), method=method),
            ax=ax,
            truncate_mode="lastp",
            p=30,
            link_color_func=lambda _: "#1B5299",
        )
        ax.set(
            title=f"Dendrograma · enlace {method} · hasta 30 ramas",
            xlabel="Observación o grupo agregado",
            ylabel="Distancia de enlace",
        )
        fig.tight_layout()
        return fig

    def characterize(self, result: Experiment) -> pd.DataFrame:
        """Provide characterize using validated inputs."""
        columns = list(self._df.select_dtypes(include=np.number).columns)
        return result.predictions.groupby("cluster")[columns].agg(["count", "mean", "std"])

    def cluster_sizes(self, result: Experiment) -> Any:
        """Inspect actual group sizes, with noise labelled separately."""
        counts = (
            result.predictions["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("cluster")
            .reset_index(name="records")
        )
        counts["cluster"] = counts["cluster"].map(
            lambda value: "Ruido" if value == -1 else f"Grupo {value}"
        )
        return style(
            px.bar(
                counts,
                x="cluster",
                y="records",
                color="cluster",
                pattern_shape="cluster",
                color_discrete_map=categorical_colors(counts["cluster"], max_categories=24),
                labels={"cluster": "Grupo", "records": "Registros"},
            ),
            "Distribución de tamaños · incluye ruido",
        )

    def profile_heatmap(self, result: Experiment) -> Any:
        """Visualize existing group means, retaining original feature units."""
        means = self.characterize(result).xs("mean", axis=1, level=1)
        if means.empty:
            raise FrameworkError("No hay variables numéricas para caracterizar grupos.")
        return style(
            px.imshow(
                means,
                text_auto=".3g",
                aspect="auto",
                color_continuous_scale=[[0, "#F1ECCE"], [1, "#1B5299"]],
                labels={"color": "Media original", "x": "Variable", "y": "Grupo"},
            ),
            "Perfil de grupos · medias en unidades originales",
        )

    def inspect_cluster(self, result: Experiment, label: int) -> pd.DataFrame:
        """Return a bounded copy of the selected original observations."""
        return result.predictions.loc[result.predictions["cluster"] == label].head(200).copy()

    def representatives(self, result: Experiment) -> pd.DataFrame:
        """Provide representatives using validated inputs."""
        x = pd.DataFrame(result.details["x"])
        x["cluster"] = result.details["labels"]
        return x[x.cluster != -1].groupby("cluster").mean()

    def compare(self) -> pd.DataFrame:
        """Compare recorded experiments only when their problem and cohort agree."""
        return compare(self.experiments)
