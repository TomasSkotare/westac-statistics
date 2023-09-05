from IPython.display import display
import pandas as pd
from ipywidgets import Dropdown, HBox, Output, ToggleButton, VBox
import matplotlib.pyplot as plt
import plotly.express as px


pd.options.mode.chained_assignment = None


class CaseOneGUI:
    PARTY_COLORS = {
        "C": "rgba(0,   153,  51,0)",
        "FI": "rgba(205,  27, 104,0)",
        "KD": "rgba(0,     0, 119,0)",
        "L": "rgba(0,   106, 179,0)",
        "MP": "rgba(131, 207,  57,0)",
        "M": "rgba(82,  189, 236,0)",
        "PP": "rgba(87,   43, 133,0)",
        "S": "rgba(232,  17,  45,0)",
        "SD": "rgba(221, 221,   0,0)",
        "V": "rgba(218,  41,  28,0)",
        "NUD": "rgba(34,   34,  34,0)",
        "gov": "rgba(127, 127, 127,0)",
        "partilös": "rgba(212, 212, 212,0)",
        "lib": "rgba(99,   99,  99,0)",
    }

    PARTY_COLORS_HEX = {
        "C": "#009933",
        "FI": "#CD1B68",
        "KD": "#000077",
        "L": "#006AB3",
        "MP": "#83CF39",
        "M": "#52BDEC",
        "PP": "#572B85",
        "S": "#E8112d",
        "SD": "#DDDD00",
        "V": "#DA291C",
        "NUD": "#222222",
        "gov": "#7F7F7F",
        "partilös": "#D4D4D4",
        "lib": "#636363",
    }

    def __init__(
        self,
        speech_df: pd.DataFrame,
        filter_key="party_abbrev",
        filter_sub_key="gender",
    ):
        self.filter_key: str = filter_key
        self.filter_sub_key: str = filter_sub_key
        self.filter_values = Dropdown(
            description="Filter", options=[], layout={"width": "160px"}
        )
        self.mode = Dropdown(
            description="Mode",
            options=["token", "speech", "speaker"],
            value="token",
            layout={"width": "160px"},
        )
        self.period = Dropdown(
            description="Period",
            options=["year", "decade"],
            value="decade",
            layout={"width": "160px"},
        )
        self.kind = Dropdown(
            description="Kind",
            options=["area", "line", "bar", "table", "excel"],
            value="table",
            layout={"width": "160px"},
        )
        self.normalize = ToggleButton(
            description="Normalize", value=True, layout={"width": "160px"}
        )
        self.output = Output()
        self.speech_df = speech_df

    def layout(self):
        return VBox(
            [
                HBox(
                    [
                        VBox([self.filter_values, self.period]),
                        VBox([self.kind, self.mode]),
                        self.normalize,
                    ]
                ),
                self.output,
            ]
        )

    def setup(self, filter_values) -> "CaseOneGUI":
        self.filter_values.options = filter_values

        self.filter_values.observe(self.handler, "value")
        self.mode.observe(self.handler, "value")
        self.period.observe(self.handler, "value")
        self.kind.observe(self.handler, "value")
        self.normalize.observe(self.handler, "value")
        return self

    def update(self):

        opts: dict = dict(
            temporal_key=self.period.value,
            filter_key=self.filter_key,
            filter_sub_key=self.filter_sub_key,
            filter_value=self.filter_values.value,
            normalize=self.normalize.value,
            mode=self.mode.value,
        )

        self.output.clear_output()
        with self.output:
            # print(opts)
            data: pd.DataFrame = self.compute_statistics(**opts)
            self.plot_filter(data, kind=self.kind.value)

    def handler(self, *_):
        self.update()

    def plot_filter(self, data, kind: str):
        if kind == "table":
            display(data.round(2))
        elif kind == "excel":
            display(data.round(2))
            data.to_excel("output.xlsx")
            print("Saved as output.xlsx")
        else:
            plot_type = None
            post_create_fun = None
            plot_common_options = {
                "x": data.index,
                "y": data.columns,
                "width": 1200,
                "height": 600,
                "color_discrete_map": CaseOneGUI.PARTY_COLORS_HEX,
            }
            plot_common_layout = {"bargap": 0.2, "yaxis_title": "Count"}
            plot_common_traces = {
                "marker_line_color": "#000000",
                "marker_line_width": 0.5,
            }

            if kind == "bar":
                plot_type = px.bar
                plot_common_options["barmode"] = "stack"
            elif kind == "line":
                plot_type = px.line
            elif kind == "area":
                plot_type = px.area
                # See https://stackoverflow.com/a/66521215 for this fix
                def area_fun(p):
                    p.for_each_trace(
                        lambda trace: trace.update(fillcolor=trace.line.color)
                    )
                    p.update_traces(line=dict(color="rgba(0,0,0,1)", width=0.0))

                post_create_fun = area_fun
                # post_create_fun = lambda p: p.for_each_trace(
                #     lambda trace: trace.update(fillcolor=trace.line.color)
                # )

            if plot_type is None:
                data.plot(kind=kind, figsize=(20, 10))
                plt.show()
            else:
                plot = plot_type(data, **plot_common_options)
                plot.update_traces(**plot_common_traces)
                plot.update_layout(**plot_common_layout)
                if post_create_fun is not None:
                    post_create_fun(plot)
                display(plot)

    def compute_statistics(
        self,
        *,
        temporal_key: str,
        filter_key: str,
        filter_sub_key: str,
        filter_value: str,
        normalize: bool,
        mode: str
    ):

        data: pd.DataFrame = self.speech_df.copy()

        if filter_value:
            data = data[data[filter_key] == filter_value]
            filter_key = filter_sub_key

        if temporal_key == "decade":
            data[temporal_key] = data.year - data.year % 10

        pivot: pd.DataFrame = None

        if mode == "token":
            pivot = (
                data.groupby([temporal_key, filter_key])
                .agg({"n_tokens": sum})
                .unstack(level=1)
            )
        elif mode == "speech":
            pivot = pd.DataFrame(
                data.groupby([temporal_key, filter_key]).size()
            ).unstack(level=1)
        elif mode == "speaker":
            pivot = (
                data.groupby([temporal_key, filter_key])
                .agg({"who": lambda x: len(set(x))})
                .unstack(level=1)
            )
        pivot = pivot.fillna(0)

        if normalize:
            pivot = pivot.div(pivot.sum(axis=1), axis=0)
        if hasattr(pivot.columns, "levels"):
            pivot.columns = pivot.columns.levels[1].tolist()

        return pivot
