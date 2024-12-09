import altair as alt
import polars as pl
import streamlit as st

import nisapi


def widget_filter(
    df: pl.DataFrame,
    column: str,
    options=None,
    default: str = None,
    n_radio_max: int = 5,
) -> pl.DataFrame:
    if options is None:
        options = df[column].unique().to_list()

    if len(options) <= n_radio_max:
        if default is not None and default in options:
            options = [default] + sorted([x for x in options if x != default])

        value = st.radio(label=column, options=options)
    else:
        if default is not None and default in options:
            index = options.index(default)
        else:
            index = 0

        value = st.selectbox(label=column, options=options, index=index)

    return df.filter(pl.col(column) == pl.lit(value)).drop(column)


if __name__ == "__main__":
    st.title("Locally cached NIS data")

    nis = nisapi.get_nis().collect()

    data = (
        nis.pipe(widget_filter, "vaccine", default="flu")
        .pipe(
            widget_filter,
            "geographic_type",
            default="nation",
            options=["nation", "region", "admin1", "substate"],
        )
        .pipe(widget_filter, "geographic_value")
        .pipe(widget_filter, "demographic_type", default="age")
        .pipe(widget_filter, "demographic_value", default="18+ years")
        .pipe(widget_filter, "time_type", default="week")
        .pipe(widget_filter, "indicator_type", default="4-level vaccination and intent")
        .pipe(widget_filter, "indicator_value", default="received a vaccination")
    )

    datasets = data["id"].unique().to_list()

    st.text(f"Showing data from datasets: {datasets}")

    time_axis = alt.Axis(format="%Y-%b-%d", tickCount="month")
    points = (
        alt.Chart(data)
        .mark_point()
        .encode(x=alt.X("time_end", axis=time_axis), y="estimate")
    )
    error_bars = (
        alt.Chart(data)
        .mark_errorbar()
        .encode(
            x=alt.X("time_end", axis=time_axis),
            y=alt.Y("lci", title="uptake", scale=alt.Scale(domain=[0.0, 1.0])),
            y2="uci",
        )
    )

    chart = points + error_bars

    st.altair_chart(chart.interactive(), use_container_width=True)
