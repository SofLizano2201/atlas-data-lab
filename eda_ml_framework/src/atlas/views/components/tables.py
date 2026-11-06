"""One presentation adapter for native, sortable and downloadable Streamlit tables."""

from typing import Any

import pandas as pd
import streamlit as st


def prepare_table(data: Any) -> Any:
    """Style modest tables without changing values, order, indices or large-table performance."""
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data
    if data.size > 20000 or not data.index.is_unique or not data.columns.is_unique:
        return data
    backgrounds = pd.DataFrame(
        [
            [
                "background-color: #FFFFFF; color: #331832"
                if row % 2 == 0
                else "background-color: #F1F5F9; color: #331832"
            ]
            * len(data.columns)
            for row in range(len(data))
        ],
        index=data.index,
        columns=data.columns,
    )
    formats = {
        column: lambda value: f"{value:.8g}"
        for column in data.columns
        if pd.api.types.is_float_dtype(data[column])
    }
    return data.style.apply(lambda _: backgrounds, axis=None).format(formats, na_rep="—")


def data_table(data: Any = None, *, target=None, **kwargs: Any) -> Any:
    """Keep Streamlit selection, column configuration and native controls intact."""
    return (target if target is not None else st).dataframe(prepare_table(data), **kwargs)
