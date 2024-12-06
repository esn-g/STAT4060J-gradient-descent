from shiny import App, ui, render, reactive
import pandas as pd
import plotly.graph_objs as go

# Sample Data
datasets = {
    "Dataset 1": pd.DataFrame({"time": range(250), "value": range(250)}),
    "Dataset 2": pd.DataFrame({"time": range(250), "value": [x**2 for x in range(250)]}),
    "Dataset 3": pd.DataFrame({"time": range(250), "value": [x**3 for x in range(250)]}),
    "Dataset 4": pd.DataFrame({"time": range(250), "value": [x**0.5 for x in range(250)]}),
    "Dataset 5": pd.DataFrame({"time": range(250), "value": [x * 2 for x in range(250)]}),
    "Dataset 6": pd.DataFrame({"time": range(250), "value": [x * 3 for x in range(250)]}),
}

static_params = pd.DataFrame({
    "Parameter": ["Param1", "Param2", "Param3"],
    "Value": [10, 20, 30],
})

app_ui = ui.page_fluid(
    # Dashboard-style header
    ui.div(
        ui.h2("A Study of Accelerated Gradient Descent for Classification Logistic Regression"),
        class_="bg-primary text-white p-3"
    ),
    # A row that mimics a sidebar and main content
    ui.row(
        ui.column(
            3,
            ui.input_checkbox_group(
                "datasets", "Select Datasets to Plot",
                {key: key for key in datasets.keys()},
                selected=list(datasets.keys())
            ),
            ui.input_slider("xmin", "X Axis Min", min=0, max=250, value=0),
            ui.input_slider("xmax", "X Axis Max", min=0, max=250, value=250),
            ui.input_slider("ymin", "Y Axis Min", min=0, max=110, value=0),
            ui.input_slider("ymax", "Y Axis Max", min=0, max=110, value=110),
            ui.h2("Static Parameters Table"),
            ui.output_table("static_table")
        ),
        ui.column(
            6,
            ui.output_plot("combined_plot")
        ),
        ui.column(
            3,
            ui.h2("Reactive Results Table"),
            ui.output_table("results_table")
        )
    )
    
)

def server(input, output, session):
    @reactive.Calc
    def valid_ranges():
        return (input.xmin() < input.xmax()) and (input.ymin() < input.ymax())

    @reactive.Effect
    def show_warning_if_invalid():
        if not valid_ranges():
            ui.notification_show(
                "Invalid axis range provided. Reverting to default (X: 0–250, Y: 0–110).",
                type="warning",
                duration=2
            )

    @output
    @render.ui
    def combined_plot_ui():
        # Determine axis limits
        if valid_ranges():
            x_min, x_max = input.xmin(), input.xmax()
            y_min, y_max = input.ymin(), input.ymax()
        else:
            # Default values if ranges are invalid
            x_min, x_max = 0, 250
            y_min, y_max = 0, 110

        selected = input.datasets()
        fig = go.Figure()

        for name, df in datasets.items():
            if name in selected:
                fig.add_trace(go.Scatter(
                    x=df["time"], y=df["value"], mode='lines', name=name
                ))

        fig.update_layout(
            title="Combined Dataset Plot",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])
        return fig

    @output
    @render.table
    def static_table():
        return static_params

    @output
    @render.table
    def results_table():
        selected = input.datasets()
        results = []
        for name in selected:
            df = datasets[name]
            results.append({
                "Dataset": name,
                "Min Time": df["time"].min(),
                "Max Time": df["time"].max(),
                "Mean Value": df["value"].mean()
            })
        return pd.DataFrame(results)

app = App(app_ui, server)
