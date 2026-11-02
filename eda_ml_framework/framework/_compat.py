"""Explicit academic aliases; no duplicated calculations."""

from functools import wraps
from inspect import signature

from atlas.models import (
    Classification,
    Clustering,
    ExploratoryDataAnalysis,
    Regression,
    SupervisedLearning,
    UnsupervisedLearning,
)

METHOD_ALIASES = {
    "historial": "history",
    "versiones": "version_count",
    "cargar": "load",
    "cargar_csv": "load_csv",
    "cargar_excel": "load_excel",
    "cargar_json": "load_json",
    "cargar_parquet": "load_parquet",
    "vista_previa": "preview",
    "dimensiones": "dimensions",
    "columnas": "columns",
    "tipos_datos": "data_types",
    "memoria": "memory_usage",
    "informacion": "information",
    "valores_unicos": "unique_values",
    "constantes": "constant_columns",
    "duplicados": "duplicates",
    "copia_seguridad": "backup",
    "aplicar": "apply",
    "restaurar": "restore",
    "exportar": "export",
    "estadisticas": "statistics",
    "estadistica": "statistic",
    "media": "mean",
    "mediana": "median",
    "moda": "mode",
    "minimo": "minimum",
    "maximo": "maximum",
    "rango": "value_range",
    "varianza": "variance",
    "desviacion_estandar": "standard_deviation",
    "cuartiles": "quartiles",
    "rango_intercuartilico": "interquartile_range",
    "coeficiente_variacion": "coefficient_of_variation",
    "asimetria": "skewness",
    "curtosis": "kurtosis",
    "percentiles": "percentiles",
    "frecuencias": "frequencies",
    "contingencia": "contingency",
    "resumen": "summary",
    "estadisticas_agrupadas": "grouped_statistics",
    "nulos": "missing_values",
    "inconsistencias": "type_inconsistencies",
    "categorias_raras": "rare_categories",
    "transformar": "transform",
    "transformar_personalizado": "custom_transform",
    "correlacion": "correlation",
    "prueba": "statistical_test",
    "grafico": "chart",
    "reporte": "report",
    "preparar": "prepare",
    "reducir": "reduce_dimensions",
    "distancias": "distances",
    "visualizar_representacion": "plot_representation",
    "comparar_representaciones": "compare_representations",
    "algoritmos": "algorithms",
    "entrenar": "train",
    "codo": "elbow",
    "dendrograma": "dendrogram",
    "caracterizar": "characterize",
    "representantes": "representatives",
    "comparar": "compare",
    "identificar_problema": "identify_problem",
    "crear_estimador": "create_estimator",
    "evaluar": "evaluate",
    "predecir": "predict",
    "importancia": "feature_importance",
    "coeficientes": "coefficients",
    "curva_aprendizaje": "learning_curve",
    "grafico_resultado": "plot_result",
    "comparar_lasso": "compare_lasso",
    "comparar_ridge": "compare_ridge",
    "comparar_svm": "compare_svm",
}
PARAMETER_ALIASES = {
    "historial": "history",
    "versiones": "version_count",
    "cargar": "load",
    "cargar_csv": "load_csv",
    "cargar_excel": "load_excel",
    "cargar_json": "load_json",
    "cargar_parquet": "load_parquet",
    "vista_previa": "preview",
    "dimensiones": "dimensions",
    "columnas": "columns",
    "tipos_datos": "data_types",
    "memoria": "memory_usage",
    "informacion": "information",
    "valores_unicos": "unique_values",
    "constantes": "constant_columns",
    "duplicados": "duplicates",
    "copia_seguridad": "backup",
    "aplicar": "apply",
    "restaurar": "restore",
    "exportar": "export",
    "estadisticas": "statistics",
    "estadistica": "statistic",
    "media": "mean",
    "mediana": "median",
    "moda": "mode",
    "minimo": "minimum",
    "maximo": "maximum",
    "rango": "value_range",
    "varianza": "variance",
    "desviacion_estandar": "standard_deviation",
    "cuartiles": "quartiles",
    "rango_intercuartilico": "interquartile_range",
    "coeficiente_variacion": "coefficient_of_variation",
    "asimetria": "skewness",
    "curtosis": "kurtosis",
    "percentiles": "percentiles",
    "frecuencias": "frequencies",
    "contingencia": "contingency",
    "resumen": "summary",
    "estadisticas_agrupadas": "grouped_statistics",
    "nulos": "missing_values",
    "inconsistencias": "type_inconsistencies",
    "categorias_raras": "rare_categories",
    "transformar": "transform",
    "transformar_personalizado": "custom_transform",
    "correlacion": "correlation",
    "prueba": "statistical_test",
    "grafico": "chart",
    "reporte": "report",
    "preparar": "prepare",
    "reducir": "reduce_dimensions",
    "distancias": "distances",
    "visualizar_representacion": "plot_representation",
    "comparar_representaciones": "compare_representations",
    "algoritmos": "algorithms",
    "entrenar": "train",
    "codo": "elbow",
    "dendrograma": "dendrogram",
    "caracterizar": "characterize",
    "representantes": "representatives",
    "comparar": "compare",
    "identificar_problema": "identify_problem",
    "crear_estimador": "create_estimator",
    "evaluar": "evaluate",
    "predecir": "predict",
    "importancia": "feature_importance",
    "coeficientes": "coefficients",
    "curva_aprendizaje": "learning_curve",
    "grafico_resultado": "plot_result",
    "comparar_lasso": "compare_lasso",
    "comparar_ridge": "compare_ridge",
    "comparar_svm": "compare_svm",
    "formato": "file_format",
    "separador": "separator",
    "codificacion": "text_encoding",
    "hoja": "sheet",
    "columna": "column",
    "nuevo": "new_frame",
    "operacion": "operation",
    "confirmar": "confirm",
    "numericas": "numeric_only",
    "nombre": "name",
    "valores": "values",
    "categoria": "category",
    "metodo": "method",
    "umbral": "threshold",
    "funcion": "function",
    "tipo": "kind",
    "algoritmo": "algorithm",
    "objetivo": "target",
    "predictores": "predictors",
    "parametros": "parameters",
    "busqueda": "search",
    "tipo_busqueda": "search_type",
    "resultado": "result",
    "datos": "data",
    "etiquetas": "labels",
    "metrica": "metric",
    "experimentos": "experiments",
}


def install_aliases():
    for cls in [
        ExploratoryDataAnalysis,
        UnsupervisedLearning,
        SupervisedLearning,
        Clustering,
        Classification,
        Regression,
    ]:
        for old, new in METHOD_ALIASES.items():
            if old == new or new not in cls.__dict__:
                continue
            descriptor = cls.__dict__[new]
            if isinstance(descriptor, property):
                setattr(cls, old, descriptor)
                continue
            function = (
                descriptor.__func__
                if isinstance(descriptor, (staticmethod, classmethod))
                else descriptor
            )
            accepted = signature(function).parameters

            def adapt(
                function=function,
                accepted=accepted,
                translate_all=old in {"entrenar", "cargar_csv"},
            ):
                @wraps(function)
                def legacy(*args, **kwargs):
                    translated = {
                        PARAMETER_ALIASES.get(k, k)
                        if translate_all or PARAMETER_ALIASES.get(k, k) in accepted
                        else k: v
                        for k, v in kwargs.items()
                    }
                    return function(*args, **translated)

                return legacy

            wrapped = adapt()
            setattr(
                cls,
                old,
                classmethod(wrapped)
                if isinstance(descriptor, classmethod)
                else staticmethod(wrapped)
                if isinstance(descriptor, staticmethod)
                else wrapped,
            )
        if hasattr(cls, "__init__") and cls in [SupervisedLearning, Clustering]:
            cls.experimentos = property(lambda self: self.experiments)
