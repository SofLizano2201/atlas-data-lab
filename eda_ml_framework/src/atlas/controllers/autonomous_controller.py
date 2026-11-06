"""Execute agent plans through existing scientific controllers, once per dataset and goal."""

from dataclasses import dataclass, field
from html import escape
from time import monotonic
from typing import Any

import pandas as pd

from atlas.controllers.agent_controller import AgentController
from atlas.controllers.automatic_eda_controller import AutomaticEDAController
from atlas.controllers.classification_controller import ClassificationController
from atlas.controllers.clustering_controller import ClusteringController
from atlas.controllers.comparison_controller import ComparisonController
from atlas.controllers.regression_controller import RegressionController
from atlas.core.dependencies import build_offline_agent
from atlas.models.experiment import TABLE_CSS
from atlas.services.experiment_service import ExperimentService


@dataclass
class AutonomousRun:
    key: str
    status: str = "running"
    target: str | None = None
    provisional: bool = True
    decisions: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    figures: dict[str, Any] = field(default_factory=dict)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0
    report: str = ""


class AutonomousController:
    @staticmethod
    def key(state, target=None):
        return f"{AutomaticEDAController.key(state['eda'])}:{target or 'auto'}"

    @staticmethod
    def run(state, target=None, progress=None, max_seconds=90):
        """Execute bounded local experiments without altering the original or working dataset."""
        key = AutonomousController.key(state, target)
        runs = state.setdefault("autonomous_runs", {})
        if key in runs:
            state["autonomous_active"] = key
            return runs[key]
        run = AutonomousRun(key)
        runs[key] = run
        state["autonomous_active"] = key
        started = monotonic()
        context = AgentController.context_for(state)
        agents = {
            name: build_offline_agent(name, context)[0]
            for name in (
                "EDAAgent",
                "ClusteringAgent",
                "ClassificationAgent",
                "RegressionAgent",
                "ModelComparisonAgent",
            )
        }

        def announce(text):
            if progress:
                progress(text)

        def available():
            if monotonic() - started >= max_seconds:
                if (
                    "Se alcanzó el presupuesto de tiempo; se conservaron los resultados completados."
                    not in run.warnings
                ):
                    run.warnings.append(
                        "Se alcanzó el presupuesto de tiempo; se conservaron los resultados completados."
                    )
                return False
            return True

        def record(result, engine):
            result.config["autonomous_run"] = key
            result.config["provisional_target"] = run.provisional
            ExperimentService.remember(state, result, engine)
            run.experiments.append(result.id)

        def attempt(label, operation):
            if not available():
                return None
            announce(label)
            try:
                return operation()
            except (
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                ImportError,
                ArithmeticError,
            ) as exc:
                run.errors.append(f"{label}: {exc}")
                return None

        announce("Comprendiendo tus datos y preparando los gráficos iniciales…")
        eda = AutomaticEDAController.run(state)
        run.figures.update(
            {
                name: item["value"]
                for name, item in eda.operations.items()
                if name.startswith("chart:") and item["status"] == "completed"
            }
        )
        frame = state["eda"].df
        evidence = {
            "columns": eda.columns,
            "suggested": eda.roles.get("suggested", []),
            "numeric": list(frame.select_dtypes(include="number")),
            "target": target,
        }
        if target is not None and target not in frame:
            run.status = "failed"
            run.errors.append("La columna que quieres predecir ya no está disponible.")
            run.report = AutonomousController.report(run, state)
            return run
        decision = agents["EDAAgent"].autonomous_plan(evidence)
        run.target, run.provisional = decision["target"], decision["provisional"]
        run.decisions.append({"agent": "EDAAgent", **decision})
        run.decisions.append(
            {
                "agent": "EDAAgent",
                "reason": "Variables excluidas por su función: "
                + ", ".join(
                    eda.roles.get("identifiers", [])
                    + eda.roles.get("dates", [])
                    + eda.roles.get("constants", [])
                )
                if any(eda.roles.get(role) for role in ("identifiers", "dates", "constants"))
                else "No detecté códigos, fechas sin preparar ni columnas constantes que excluir por su función.",
            }
        )
        evidence["target"] = run.target
        run.decisions.append(
            {
                "agent": "EDAAgent",
                "reason": "No elimino duplicados ni valores poco habituales automáticamente: podrían ser casos válidos. Los predictores incompletos se rellenan dentro de cada partición de entrenamiento; la variable que se predice no se imputa.",
            }
        )
        run.warnings.append(
            "La selección automática no conoce el significado del negocio, relaciones temporales ni posibles variables que revelen la respuesta. Revisa estas hipótesis antes de usar predicciones reales."
        )

        # Identical bounded row selections and feature sets within each comparison.
        if run.target and len(frame) >= 20:
            metadata = next(item for item in eda.columns if item["name"] == run.target)
            classification = run.target not in evidence["numeric"] or metadata["unique"] <= 20
            name = "ClassificationAgent" if classification else "RegressionAgent"
            run.decisions.append(
                {
                    "agent": "RegressionAgent" if classification else "ClassificationAgent",
                    "reason": "No entreno predicciones numéricas para este objetivo porque lo estoy explorando como categorías."
                    if classification
                    else "No entreno clasificación para este objetivo porque lo estoy explorando como una cantidad. Convertir cada valor en una clase daría una comparación sin sentido.",
                }
            )
            cls = ClassificationController if classification else RegressionController
            plan = agents[name].autonomous_plan(evidence)
            run.decisions.append({"agent": name, **plan})
            subset = (
                frame
                if len(frame) <= 3000
                else ClusteringController.sample_for_analysis(frame, 3000, 42)
            )
            if len(subset) != len(frame):
                run.warnings.append(
                    "Predicción: selección reproducible de 3.000 registros; los resultados describen esa selección, no todo el dataset."
                )
            if metadata["missing_percentage"]:
                run.warnings.append(
                    "No entrené predicciones porque la columna objetivo tiene información faltante. Decide qué hacer con esos registros en Preparación."
                )
            elif not plan["features"]:
                run.warnings.append("No hay predictores compatibles para entrenar una predicción.")
            else:
                supervised = []
                for algorithm in plan["algorithms"]:
                    engine = cls(subset, random_state=42)
                    parameters = {"class_weight": "balanced"} if classification else {}
                    if algorithm == "tree":
                        parameters["max_depth"] = 5
                    if algorithm == "forest":
                        parameters.update(n_estimators=40, max_depth=8)
                    result = attempt(
                        f"Probando {algorithm} para predecir «{run.target}»…",
                        lambda engine=engine, algorithm=algorithm, parameters=parameters: (
                            engine.train(
                                algorithm,
                                run.target,
                                plan["features"],
                                cv=3,
                                test_size=0.25,
                                scaler="standard",
                                encoding="onehot",
                                parameters=parameters,
                            )
                        ),
                    )
                    if result is not None:
                        record(result, engine)
                        supervised.append(result)
                AutonomousController.compare(
                    run, supervised, plan, agents["ModelComparisonAgent"], state
                )
                if supervised:
                    best_id = run.comparisons[-1]["best"] if run.comparisons else supervised[0].id
                    best = next(item for item in supervised if item.id == best_id)
                    engine = state["models"][best.id]
                    kinds = (
                        ["confusion", "roc"]
                        if classification
                        else ["actual_predicted", "residuals"]
                    )
                    for kind in kinds:
                        graph = attempt(
                            f"Preparando el gráfico de {kind}…",
                            lambda kind=kind: engine.plot_result(best, kind),
                        )
                        if graph is not None:
                            run.figures[kind] = graph
        else:
            run.warnings.append(
                "La predicción supervisada necesita al menos 20 registros y una columna objetivo compatible."
            )

        plan = agents["ClusteringAgent"].autonomous_plan(evidence)
        run.decisions.append({"agent": "ClusteringAgent", **plan})
        if plan["features"] and len(frame) >= 6:
            subset = (
                frame
                if len(frame) <= 1000
                else ClusteringController.sample_for_analysis(frame, 1000, 42)
            )
            if len(subset) != len(frame):
                run.warnings.append(
                    "Agrupamiento: selección reproducible de 1.000 registros; las etiquetas solo corresponden a esos registros."
                )
            clustered = []
            for algorithm, k in plan["trials"]:
                engine = ClusteringController(subset, random_state=42)
                result = attempt(
                    f"Buscando {k} grupos con {algorithm}…",
                    lambda engine=engine, algorithm=algorithm, k=k: engine.train(
                        algorithm,
                        plan["features"],
                        scaler="standard",
                        encoding="onehot",
                        n_clusters=k,
                    ),
                )
                if result is not None:
                    record(result, engine)
                    clustered.append(result)
            AutonomousController.compare(
                run, clustered, plan, agents["ModelComparisonAgent"], state
            )
            if clustered:
                best_id = (
                    run.comparisons[-1]["best"]
                    if run.comparisons and run.comparisons[-1]["problem"] == "clustering"
                    else clustered[0].id
                )
                best = next(item for item in clustered if item.id == best_id)
                engine = state["models"][best.id]
                run.figures["cluster_sizes"] = engine.cluster_sizes(best)
                if len(plan["features"]) >= 2:
                    reduced = attempt(
                        "Dibujando los grupos encontrados…",
                        lambda: engine.reduce_dimensions(
                            plan["features"], dimensions=2, scaler="standard", encoding="onehot"
                        ),
                    )
                    if reduced is not None:
                        run.figures["clusters"] = engine.plot_representation(
                            reduced["datos"], best.details["labels"]
                        )
        else:
            run.warnings.append(
                "No hay suficientes registros o características compatibles para explorar grupos."
            )
        run.seconds = monotonic() - started
        run.status = (
            "partial"
            if run.errors or not run.experiments or monotonic() - started >= max_seconds
            else "completed"
        )
        announce("Generando el informe con decisiones, resultados y gráficos…")
        run.report = AutonomousController.report(run, state)
        return run

    @staticmethod
    def compare(run, results, plan, agent, state):
        if not results:
            return
        try:
            table, best, value = ComparisonController.rank(
                results, plan["metric"], plan["direction"]
            )
            run.comparisons.append(
                {
                    "problem": results[0].problem,
                    "metric": plan["metric"],
                    "best": best,
                    "value": value,
                    "table": table,
                }
            )
            run.decisions.append(
                {
                    "agent": agent.name,
                    **agent.autonomous_plan({}),
                    "reason": f"Recomendación entre {len(results)} alternativas según {plan['metric']}: {next(item.name for item in results if item.id == best)}. No es un ganador universal.",
                }
            )
            run.figures[f"comparison_{results[0].problem}"] = ComparisonController.chart(
                results, plan["metric"], direction=plan["direction"]
            )
        except (ValueError, TypeError, KeyError) as exc:
            run.errors.append(f"Comparación no disponible: {exc}")

    @staticmethod
    def report(run, state):
        """Build a self-contained report from stored outputs only, escaping all data text."""
        sections = [
            "<h1>ATLAS · Análisis automático</h1>",
            f"<p>Dataset: {escape(str(state.get('dataset_name', 'Datos')))} · Objetivo: {escape(str(run.target))}</p>",
            "<p>El objetivo automático es provisional. La recomendación se limita a las alternativas evaluadas.</p>",
            "<h2>Decisiones y variables</h2>",
        ]
        for item in run.decisions:
            sections.append(f"<h3>{escape(item['agent'])}</h3><p>{escape(item['reason'])}</p>")
            if item.get("features"):
                sections.append(f"<p>Variables: {escape(', '.join(item['features']))}</p>")
        for comparison in run.comparisons:
            sections.append(
                f"<h2>{escape(comparison['problem'])} · {escape(comparison['metric'])}</h2>"
            )
            sections.append(comparison["table"].to_html(escape=True))
        sections.append("<h2>Calidad y exploración de los datos</h2>")
        execution = state.get("automatic_eda", {}).get(AutomaticEDAController.key(state["eda"]))
        if execution:
            sections.append(pd.DataFrame([execution.profile]).to_html(index=False, escape=True))
            sections.append(pd.DataFrame(execution.columns).to_html(index=False, escape=True))
            for operation, item in execution.operations.items():
                if isinstance(item.get("value"), pd.DataFrame):
                    sections.append(
                        f"<h3>{escape(operation)}</h3>"
                        + item["value"].head(50).to_html(escape=True)
                    )
            sections.append(
                "<p>Las tablas exploratorias se limitan a las primeras 50 filas de cada resultado para mantener el informe manejable.</p>"
            )
        sections.append("<h2>Experimentos y reproducibilidad</h2>")
        for result in state.get("experiments", []):
            if result.id not in run.experiments:
                continue
            sections.append(f"<h3>{escape(result.name)} · {escape(result.problem)}</h3>")
            sections.append(pd.Series(result.metrics, name="valor").to_frame().to_html(escape=True))
            sections.append(f"<pre>{escape(str(result.config))}</pre>")
            sections.append(
                "<p>Primeras 100 predicciones o etiquetas:</p>"
                + result.predictions.head(100).to_html(escape=True)
            )
        sections.append("<h2>Advertencias y operaciones no completadas</h2>")
        sections.extend(f"<p>{escape(message)}</p>" for message in run.warnings + run.errors)
        sections.append("<h2>Gráficos calculados</h2>")
        for index, (name, graph) in enumerate(run.figures.items()):
            sections.append(f"<h3>{escape(name)}</h3>")
            sections.append(graph.to_html(full_html=False, include_plotlyjs=index == 0))
        return (
            '<!doctype html><html lang="es"><meta charset="utf-8"><title>Informe ATLAS</title><style>body{font:15px Arial;background:#F1ECCE;color:#331832;margin:32px}h1,h2{color:#1B5299}'
            + TABLE_CSS
            + "</style><body>"
            + "".join(sections)
            + "</body></html>"
        )
