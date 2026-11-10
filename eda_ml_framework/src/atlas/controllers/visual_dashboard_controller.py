"""Reuse existing EDA charts and cache only a bounded visual working set."""

from atlas.controllers.eda_controller import EDAController


class VisualDashboardController:
    @staticmethod
    def chart(state, kind, **parameters):
        eda = state["eda"]
        key = (eda.dataset_version.content_hash, kind, repr(sorted(parameters.items())))
        cache = state.setdefault("visual_dashboard_cache", {})
        if key not in cache:
            cache[key] = EDAController(eda).chart(kind, **parameters)
            while len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[key]
