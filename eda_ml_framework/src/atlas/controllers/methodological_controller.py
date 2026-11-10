"""Session-scoped, bounded validation cache and review history."""

import json
from hashlib import sha256

from atlas.models.methodological_validation import MethodologicalValidation


class MethodologicalController:
    @staticmethod
    def correct_target_selection(state, *, confirm=False):
        """Use the existing configuration adapter only after explicit confirmation."""
        if not confirm:
            raise ValueError("Confirma explícitamente la corrección de la selección.")
        from atlas.controllers.agent_controller import AgentController
        from atlas.controllers.configuration_controller import (
            ConfigurationArgs,
            ConfigurationController,
        )

        context = AgentController.context_for(state)
        selection = dict(context.selections.get("analysis_selection") or {})
        selection["problem"] = {"clasificacion": "classification", "regresion": "regression"}.get(
            selection.get("problem"), selection.get("problem")
        )
        selection["predictors"] = [
            c for c in selection.get("predictors", []) if c != selection.get("target")
        ]
        request = ConfigurationArgs.model_validate(
            {k: v for k, v in selection.items() if k in ConfigurationArgs.model_fields}
        )
        return ConfigurationController.apply(context, request)

    @staticmethod
    def synchronize(state, config):
        eda = state["eda"]
        version = eda.dataset_version
        experiments = state.get("experiments", [])
        key = sha256(
            json.dumps(
                [
                    version.dataset_id,
                    version.version,
                    version.content_hash,
                    config,
                    state.get("navigation"),
                    [e.id for e in experiments],
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        cache = state.setdefault("methodological_cache", {})
        if key not in cache:
            run = state.get("scientific_investigations", {}).get(state.get("active_investigation"))
            if state.get("navigation") == "scientific_lab" and run:
                from atlas.models.scientific_experiment import ScientificExperiment

                cache[key] = ScientificExperiment.validate(run["definition"], run["dataset"])
            else:
                cache[key] = MethodologicalValidation.validate(eda, config)
            if state.get("navigation") in {"⚖️ Experimentos", "experiment_history"}:
                cache[key] += MethodologicalValidation.comparison(experiments)
            for experiment in experiments[-1:]:
                if experiment.dataset_id == version.content_hash:
                    cache[key] += [
                        f
                        for f in experiment.details.get("methodological_findings", [])
                        if f.finding_id in {"silhouette_domain", "split_duplicates"}
                    ]
            cache[key] = list({f.finding_id: f for f in cache[key]}.values())
            while len(cache) > 30:
                cache.pop(next(iter(cache)))
            history = state.setdefault("methodological_history", [])
            history.append((key, version.version, cache[key]))
            del history[:-100]
        state["methodological_findings"] = cache[key]
        state["methodological_key"] = key
        return cache[key]
