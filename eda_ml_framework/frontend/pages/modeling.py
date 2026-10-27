"""Entrenamiento, representación y comparación de experimentos reales."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from framework import AnalisisDatosExploratorio, Clasificacion, Clustering, Regresion
from framework.exceptions import FrameworkError
from framework.results import compare, export_frame, html_report
from framework.visualization import PALETTE, style
from frontend.components.layout import algorithm_cards, metric_cards, page_header
from frontend.components.ui import figure


def remember(result: object, engine: object) -> None:
    """Guardar los recursos por sesión, sin caché global de modelos."""
    result.model_version = len(st.session_state.experiments) + 1
    result.config["model_version"] = result.model_version
    st.session_state.experiments.append(result)
    st.session_state.models[result.id] = engine
    st.session_state.metrics[result.id] = result.metrics


def parameters(algorithm: str, problem: str) -> dict:
    """Controles guiados y parámetros avanzados JSON de datos, nunca código."""
    params = {}
    if algorithm in {"tree", "forest", "extra_trees", "gradient_boosting"}:
        params["max_depth"] = st.slider("Profundidad máxima", 1, 20, 4)
    if algorithm in {"forest", "extra_trees", "gradient_boosting", "adaboost"}:
        params["n_estimators"] = st.slider("Estimadores", 10, 300, 100)
    if algorithm == "knn":
        params["n_neighbors"] = st.slider("Vecinos", 1, 30, 5)
    if algorithm.startswith(("svm_", "svr_")):
        params["C"] = st.number_input("Regularización C", 0.01, 1000.0, 1.0)
    if algorithm in {"lasso", "ridge", "elasticnet"}:
        params["alpha"] = st.number_input("Regularización alpha", 0.0001, 100.0, 1.0)
    if problem == "clasificacion" and algorithm in {
        "logistic",
        "tree",
        "forest",
        "extra_trees",
        "svm_linear",
        "svm_rbf",
        "svm_poly",
    }:
        balanced = st.checkbox("Ponderar clases desbalanceadas", value=True)
        params["class_weight"] = "balanced" if balanced else None
    with st.expander("Parámetros avanzados"):
        st.caption(
            'Objeto JSON, por ejemplo {"min_samples_leaf": 3}. Sobrescribe los controles anteriores.'
        )
        text = st.text_area("Parámetros del estimador", "{}")
        extra = json.loads(text)
        if not isinstance(extra, dict):
            raise FrameworkError("Los parámetros deben ser un objeto JSON.")
        params.update(extra)
    return params


def supervised_page(problem: str) -> None:
    """Flujo común de clasificación/regresión con datos, variables y CV explícitos."""
    page_header("classification" if problem == "clasificacion" else "regression")
    st.session_state.problem = problem
    source = st.radio(
        "Datos de modelado", ["Original (recomendado)", "Trabajo transformado"], horizontal=True
    )
    df = st.session_state.eda.original if source.startswith("Original") else st.session_state.eda.df
    if source.startswith("Trabajo"):
        st.warning(
            "Transformaciones exploratorias ajustadas sobre todas las filas pueden sesgar "
            "la evaluación. Usa el original para estimar generalización."
        )
    cls = Clasificacion if problem == "clasificacion" else Regresion
    objective_col, predictors_col, test_col, folds_col, seed_col = st.columns([1, 1.8, 1, 0.8, 0.8])
    target = objective_col.selectbox(
        "Variable objetivo", list(df.columns), index=len(df.columns) - 1
    )
    features = predictors_col.multiselect(
        "Predictores · excluye identificadores",
        [c for c in df if c != target],
        default=[c for c in df if c != target][:12],
    )
    size = test_col.slider("Proporción de prueba", 0.05, 0.5, 0.2, 0.05)
    folds = folds_col.number_input("Folds de validación cruzada", 2, 10, 5)
    seed = seed_col.number_input("Semilla reproducible", 0, 99999, 42)
    st.session_state.selected_variables = features
    if problem == "clasificacion":
        distribution = cls(df).frecuencias(target)
        with st.expander("Distribución de clases y definición de métricas"):
            st.dataframe(distribution, use_container_width=True)
            st.caption(
                "Precision, Recall y F1 usan promedio macro; se añaden métricas binarias cuando corresponde."
            )
    with st.expander("Preprocesamiento del pipeline"):
        a, b, c = st.columns(3)
        scaler = a.selectbox("Escalamiento", ["standard", "minmax", "robust", "normalizer", "none"])
        encoding = b.selectbox("Categorías", ["onehot", "ordinal"])
        imputation = c.selectbox(
            "Imputación numérica", ["median", "mean", "most_frequent", "constant"]
        )
    st.session_state.preprocessing = {
        "scaler": scaler,
        "encoding": encoding,
        "imputation": imputation,
    }
    st.subheader(
        "Modelos de clasificación" if problem == "clasificacion" else "Modelos de regresión"
    )
    default = "logistic" if problem == "clasificacion" else "linear_multiple"
    algorithm_key = f"atlas_algorithms_{problem}"
    st.session_state.setdefault(algorithm_key, [default])
    algorithm_cards(
        [
            (
                default,
                "Regresión logística" if problem == "clasificacion" else "Lineal múltiple",
                "Modelo lineal interpretable.",
            ),
            (
                "knn" if problem == "clasificacion" else "ridge",
                "KNN" if problem == "clasificacion" else "Ridge",
                "Vecinos cercanos." if problem == "clasificacion" else "Regularización L2.",
            ),
            ("forest", "Random Forest", "Ensamble de árboles de decisión."),
            (
                "svm_rbf" if problem == "clasificacion" else "svr_rbf",
                "SVM" if problem == "clasificacion" else "SVR",
                "Máquinas de soporte con kernel RBF.",
            ),
            ("gradient_boosting", "Gradient Boosting", "Ensamble secuencial de árboles."),
        ],
        algorithm_key,
    )
    algorithms = st.multiselect(
        "Algoritmos a entrenar",
        cls.algoritmos(),
        key=algorithm_key,
    )
    with st.expander("Parámetros del modelo"):
        params = parameters(algorithms[0], problem) if len(algorithms) == 1 else {}
    search = None
    kind = "grid"
    with st.expander("Búsqueda de hiperparámetros"):
        enabled = st.checkbox("Activar búsqueda dentro de entrenamiento")
        kind = st.selectbox("Estrategia", ["grid", "random"])
        key = st.selectbox("Parámetro", ["alpha", "C", "max_depth", "n_estimators", "n_neighbors"])
        values = st.text_input("Valores separados por coma", "1, 3, 5")
        if enabled:
            candidates = [float(v.strip()) for v in values.split(",")]
            search = {
                key: [int(v) for v in candidates]
                if key in {"max_depth", "n_estimators", "n_neighbors"}
                else candidates
            }
            st.caption("Las opciones deben ser compatibles con cada algoritmo seleccionado.")
    if st.button("Entrenar y evaluar", type="primary", disabled=not algorithms):
        progress = st.progress(0, text="Preparando experimentos…")
        for i, algorithm in enumerate(algorithms):
            engine = cls(df, random_state=int(seed))
            try:
                with st.spinner(f"Entrenando {algorithm}…"):
                    result = engine.entrenar(
                        algorithm,
                        target,
                        features,
                        test_size=size,
                        cv=int(folds),
                        scaler=scaler,
                        encoding=encoding,
                        imputation=imputation,
                        parametros=params,
                        busqueda=search,
                        tipo_busqueda=kind,
                    )
                    remember(result, engine)
            except (ValueError, TypeError, ImportError, RuntimeError) as exc:
                st.error(f"{algorithm}: {exc}")
            progress.progress(
                (i + 1) / len(algorithms), text=f"{i + 1}/{len(algorithms)} completados"
            )
    results = [e for e in st.session_state.experiments if e.problem == problem]
    if not results:
        st.info("Selecciona el objetivo y los predictores para crear tu primer experimento.")
        return
    result = st.selectbox("Resultado", results, format_func=lambda e: f"{e.name} · {e.id}")
    engine = st.session_state.models[result.id]
    featured = (
        ["Accuracy", "Precision", "Recall", "F1"]
        if problem == "clasificacion"
        else ["MAE", "RMSE", "R2", "MAPE"]
    )
    available = [
        (name, f"{result.metrics[name]:.4g}", "Conjunto de prueba")
        for name in featured
        if name in result.metrics and result.metrics[name] is not None
    ]
    if available:
        metric_cards(available)
    first, second = st.columns(2)
    chart_types = (
        ["confusion", "roc"] if problem == "clasificacion" else ["actual_predicted", "residuals"]
    )
    for column, chart_type in zip([first, second], chart_types):
        with column.container(border=True):
            try:
                diagnostic = engine.grafico_resultado(result, chart_type)
                figure(diagnostic, f"summary_{result.id}_{chart_type}")
            except (ValueError, TypeError) as exc:
                st.info(f"Diagnóstico no disponible: {exc}")
    metrics, charts, predictions, advanced = st.tabs(
        ["Métricas", "Gráficos", "Predicciones", "Interpretación"]
    )
    with metrics:
        st.dataframe(pd.Series(result.metrics, name="valor").to_frame(), use_container_width=True)
        st.caption(
            "CV de regresión usa RMSE negativo (más alto es mejor). CV_seleccion es el valor "
            "usado para elegir parámetros, no una estimación independiente."
        )
        if problem == "clasificacion":
            st.dataframe(result.details["report"])
    with charts:
        kinds = (
            ["confusion", "confusion_normalized", "roc", "pr", "tree", "correctness"]
            if (problem == "clasificacion")
            else ["actual_predicted", "residuals", "histogram", "qq"]
        )
        chart = st.selectbox("Diagnóstico", kinds)
        if st.button("Mostrar diagnóstico"):
            st.session_state.graphs[result.id] = engine.grafico_resultado(result, chart)
        if result.id in st.session_state.graphs:
            figure(st.session_state.graphs[result.id], "model")
    with predictions:
        st.dataframe(result.predictions, use_container_width=True)
        st.download_button("Predicciones CSV", export_frame(result.predictions), f"{result.id}.csv")
        new_file = st.file_uploader(
            "Predecir nuevos datos (CSV UTF-8)", type=["csv"], key="predict_upload"
        )
        if new_file and st.button("Generar nuevas predicciones"):
            data = AnalisisDatosExploratorio.cargar(new_file).df
            st.session_state.new_predictions = engine.predecir(result, data)
        if "new_predictions" in st.session_state:
            st.dataframe(st.session_state.new_predictions)
            st.download_button(
                "Descargar nuevas predicciones",
                export_frame(st.session_state.new_predictions),
                "nuevas_predicciones.csv",
            )
    with advanced:
        st.caption(
            "La importancia sobre prueba sirve para interpretar; no selecciones variables con ella."
        )
        if st.button("Calcular importancia por permutación"):
            importance = engine.importancia(result)
            st.dataframe(importance)
            figure(
                style(
                    px.bar(
                        importance.sort_values("importancia"),
                        y="variable",
                        x="importancia",
                        orientation="h",
                        error_x="std",
                        color_discrete_sequence=PALETTE,
                        labels={
                            "importancia": "Cambio en puntuación al permutar",
                            "variable": "Variable",
                        },
                    ),
                    "Importancia por permutación · conjunto de prueba",
                ),
                "importance",
            )
        if st.button("Consultar coeficientes"):
            st.dataframe(engine.coeficientes(result))
        if st.button("Calcular curva de aprendizaje"):
            table = engine.curva_aprendizaje(result)
            figure(
                style(
                    px.line(
                        table,
                        x="filas",
                        y=["entrenamiento", "validacion"],
                        color_discrete_sequence=PALETTE,
                        labels={
                            "filas": "Registros de entrenamiento",
                            "value": "Puntuación",
                            "variable": "Conjunto",
                        },
                    ),
                    "Curva de aprendizaje",
                ),
                "learning",
            )
    experiment_downloads(result)


def clustering_page() -> None:
    """Representación, clustering y evaluación sobre variables elegidas."""
    page_header("clustering")
    df = st.session_state.eda.df
    st.subheader("Configuración del análisis")
    variables, scale, categories, random = st.columns([1.5, 1, 1, 0.8])
    features = variables.multiselect("Variables de agrupamiento", list(df), default=list(df)[:4])
    scaler = scale.selectbox("Escalamiento", ["standard", "minmax", "robust", "none"])
    encoding = categories.selectbox("Codificación categórica", ["onehot", "ordinal"])
    seed = random.number_input("Semilla", 0, 99999, 42)
    engine = Clustering(df, random_state=int(seed))
    train, representation = st.tabs(["Agrupamiento", "Representación avanzada"])
    with representation:
        method = st.selectbox("Reducción de dimensionalidad", ["pca", "tsne", "umap"])
        dimensions = st.radio("Dimensiones", [2, 3], horizontal=True)
        st.caption(
            "t-SNE y UMAP son exploratorios; las distancias visuales no equivalen a las originales."
        )
        if st.button("Calcular representación"):
            with st.spinner("Calculando representación…"):
                reduced = engine.reducir(
                    features, method, dimensions, scaler=scaler, encoding=encoding
                )
                st.session_state.graphs["representation"] = engine.visualizar_representacion(
                    reduced["datos"]
                )
                st.session_state.representation = reduced
        if "representation" in st.session_state.graphs:
            figure(st.session_state.graphs["representation"], "representation")
            variance = st.session_state.representation["varianza"]
            if variance is not None:
                st.write("Varianza explicada:", variance)
                components = st.session_state.representation["modelo"].components_
                st.dataframe(
                    pd.DataFrame(
                        components.T, columns=[f"PCA {i + 1}" for i in range(len(variance))]
                    )
                )
        metric = st.selectbox("Distancia", ["euclidean", "cityblock", "cosine"])
        if st.button("Calcular matriz de distancias"):
            st.dataframe(engine.distancias(features, metric))
    with train:
        st.session_state.setdefault("atlas_cluster_algorithm", "kmeans")
        algorithm_cards(
            [
                ("kmeans", "K-Means", "Grupos compactos mediante centroides."),
                ("dbscan", "DBSCAN", "Densidad y detección de ruido."),
                ("hierarchical", "Jerárquico", "Estructura mediante enlaces."),
                ("kmedoids", "K-Medoids", "Representantes; dependencia opcional."),
            ],
            "atlas_cluster_algorithm",
            multiple=False,
        )
        algorithm = st.selectbox("Algoritmo", engine.algoritmos(), key="atlas_cluster_algorithm")
        params = {}
        if algorithm not in {"dbscan", "optics"}:
            k = st.slider("Número de grupos", 2, 20, 3)
            params["n_components" if algorithm == "gaussian" else "n_clusters"] = k
        if algorithm == "dbscan":
            params["eps"] = st.number_input("Radio eps", 0.01, 20.0, 0.5)
        if algorithm in {"dbscan", "optics"}:
            params["min_samples"] = st.number_input("Mínimo de vecinos", 2, 100, 5)
        if algorithm in {"hierarchical", "agglomerative"}:
            params["method" if algorithm == "hierarchical" else "linkage"] = st.selectbox(
                "Enlace", ["ward", "complete", "average", "single"]
            )
        if st.button("Ejecutar agrupamiento", type="primary"):
            with st.spinner("Calculando grupos…"):
                result = engine.entrenar(
                    algorithm, features, scaler=scaler, encoding=encoding, **params
                )
                remember(result, engine)
        if st.button("Calcular método del codo"):
            table = engine.codo(features)
            figure(
                style(
                    px.line(
                        table,
                        x="k",
                        y="inercia",
                        markers=True,
                        color_discrete_sequence=PALETTE,
                        labels={
                            "k": "Número de grupos (K)",
                            "inercia": "Inercia · suma de distancias cuadráticas",
                        },
                    ),
                    "Método del codo · K-Means",
                ),
                "elbow",
            )
        if st.button("Crear dendrograma"):
            figure(engine.dendrograma(features), "dendrogram")
    results = [e for e in st.session_state.experiments if e.problem == "clustering"]
    if not results:
        st.info("Ejecuta un algoritmo para explorar sus grupos y métricas.")
        return
    result = st.selectbox(
        "Agrupamiento guardado", results, format_func=lambda e: f"{e.name} · {e.id}"
    )
    saved = st.session_state.models[result.id]
    metric_cards(
        [
            (name, f"{result.metrics[name]:.4g}", "Ruido excluido de los índices internos")
            for name in ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz", "ruido"]
            if name in result.metrics
        ]
    )
    st.dataframe(pd.Series(result.metrics, name="valor").to_frame())
    st.caption(
        "Ruido: cluster -1. Las métricas internas excluyen ruido; no se calculan para un solo grupo."
    )
    st.dataframe(saved.caracterizar(result))
    st.dataframe(result.predictions.cluster.value_counts().rename("tamaño").to_frame())
    with st.expander("Representantes en espacio transformado"):
        st.dataframe(saved.representantes(result))
    dimensions = st.radio("Visualización de clusters", [2, 3], horizontal=True)
    if st.button("Visualizar grupos con PCA"):
        reduced = saved.reducir(
            result.config["columnas"],
            dimensiones=dimensions,
            scaler=result.config["scaler"],
            encoding=result.config["encoding"],
        )
        figure(
            saved.visualizar_representacion(reduced["datos"], result.details["labels"]), "clusters"
        )
    if st.button("Silhouette Plot"):
        figure(saved.silhouette_plot(result), "silhouette")
    st.download_button("Exportar etiquetas CSV", export_frame(result.predictions), "clusters.csv")
    experiment_downloads(result)


def experiment_downloads(result: object) -> None:
    """Reporte, modelo y configuración de una ejecución real."""
    st.subheader("Exportar experimento")
    st.download_button(
        "Configuración JSON",
        json.dumps(result.config, indent=2, default=str),
        f"{result.id}.json",
        "application/json",
    )
    st.download_button("Modelo entrenado joblib", result.model_bytes(), f"{result.id}.joblib")
    st.caption("Exportación para uso Python de confianza; la aplicación no carga archivos joblib.")
    report = html_report(
        f"Atlas · {result.name}",
        {
            "Métricas": pd.Series(result.metrics, name="valor").to_frame(),
            "Predicciones": result.predictions.head(100),
        },
        f"Tiempo total: {result.seconds:.3f} s. Configuración: {result.config}",
    )
    st.download_button("Reporte del modelo HTML", report, f"{result.id}.html", "text/html")


def comparison_page() -> None:
    """Comparar solo cohortes compatibles, elegir métrica y sentido de optimización."""
    page_header("experiments")
    experiments = st.session_state.experiments
    if not experiments:
        st.info("Entrena modelos o crea agrupamientos para comenzar la comparación.")
        return
    selected = st.multiselect(
        "Experimentos", experiments, format_func=lambda e: f"{e.problem} · {e.name} · {e.id}"
    )
    if not selected:
        return
    table = compare(selected)
    controls, criterion = st.columns([2, 1])
    metric = controls.selectbox("Métrica a comparar", list(table.select_dtypes(include=np.number)))
    lower_default = metric in {
        "MAE",
        "MSE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "ER",
        "Log_loss",
        "Davies_Bouldin",
        "Inercia",
        "segundos",
        "ruido",
    }
    direction = criterion.radio(
        "Criterio",
        ["Menor es mejor", "Mayor es mejor"],
        index=0 if lower_default else 1,
        horizontal=True,
    )
    values = table[metric].dropna()
    results_column, recommendation = st.columns([3, 1])
    with results_column.container(border=True):
        st.subheader("Resultados de la comparación")
        st.dataframe(table, use_container_width=True)
    with recommendation.container(border=True):
        st.subheader("Recomendación según la métrica")
        if not values.empty:
            best = values.idxmin() if direction.startswith("Menor") else values.idxmax()
            st.markdown(f"**{table.loc[best, 'modelo']}**")
            st.metric(metric, f"{values.loc[best]:.4g}")
            st.caption(f"Experimento {best}")
        st.caption(
            "La conclusión depende de esta métrica, partición y contexto; no hay un ganador universal. "
            "Usar reiteradamente la prueba para seleccionar modelos sesga su evaluación final."
        )
    metric_cards(
        [
            ("Experimentos", len(selected), "Modelos seleccionados"),
            (
                "Tiempo promedio",
                f"{np.mean([e.seconds for e in selected]):.2f} s",
                "Por entrenamiento",
            ),
            ("Métrica", metric, direction),
        ]
    )
    figure(
        style(
            px.bar(
                table.reset_index(),
                x="id",
                y=metric,
                color="modelo",
                color_discrete_sequence=PALETTE,
                labels={"id": "Experimento", "modelo": "Modelo", metric: metric},
            ),
            f"Comparación de modelos · {metric}",
        ),
        "comparison",
    )
    for result in selected:
        with st.expander(f"Parámetros · {result.name} · {result.id}"):
            st.json(result.config)
    st.download_button("Comparación CSV", export_frame(table.reset_index()), "comparacion.csv")
    st.download_button(
        "Comparación Excel", export_frame(table.reset_index(), "xlsx"), "comparacion.xlsx"
    )
    st.download_button(
        "Reporte comparativo HTML",
        html_report(
            "Atlas · Comparación",
            {"Resultados": table},
            f"Métrica: {metric}; criterio: {direction}",
        ),
        "comparacion.html",
        "text/html",
    )
