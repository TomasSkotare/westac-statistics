import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from IPython.core.display import display, HTML, DisplayHandle
from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea


class Westac_Histogram:
    def __init__(self, speech_index: pd.DataFrame, title, callback: callable):
        self.speech_index = speech_index
        self.pivot_key = Dropdown(
            options={
                "Protocol name": "protocol_name",
                "Document name": "document_name",
            },
            value="protocol_name",
        )
        self.output = Output()

    def show(self):
        df = self.speech_index.groupby([self.pivot_key.value])

    def do_things(self, trace, points, state):
        with self.output:
            display(self.create_thing(trace, points, state))

        output.value = self.create_thing(trace, points, state)
        #         output.value = '\n'.join([x for x,y in df.iloc[points.point_inds].items()])

        # df = SPEECH_INDEX.set_index('document_name')['n_tokens']
        #     df = SPEECH_INDEX.groupby(['protocol_name'])['n_tokens'].sum()

        fig = px.histogram(df, x="n_tokens", nbins=100)
        fig.update_layout(
            bargap=0.2,
            title="Histogram of number of tokens per protocol",
            xaxis_title="Number of tokens",
            yaxis_title="Count",
            height=600,
            width=1000,
        )
        fig.update_traces(
            marker_color="rgb(158,202,225)",
            marker_line_color="rgb(8,48,107)",
            marker_line_width=1.5,
            opacity=0.6,
            hovertemplate="Token range = %{x}<br>Count = %{y}<extra></extra>",
        )

        output = Textarea(placeholder="None selected", layout={"height": "500px"})

        fig_widget = go.FigureWidget(fig.data, fig.layout)
        # print(dir(fig))
        if on_select is not None:
            fig_widget.data[0].on_selection(do_things)
        if on_click is not None:
            fig_widget.data[0].on_click(do_things)
