from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt

# Sample Data Preparation
datasets = {
    "Dataset 1": pd.DataFrame({"time": range(250), "value": range(250)}),
    "Dataset 2": pd.DataFrame({"time": range(250), "value": [x**2 for x in range(250)]}),
    "Dataset 3": pd.DataFrame({"time": range(250), "value": [x**3 for x in range(250)]}),
    "Dataset 4": pd.DataFrame({"time": range(250), "value": [x**0.5 for x in range(250)]}),
    "Dataset 5": pd.DataFrame({"time": range(250), "value": [x * 2 for x in range(250)]}),
    "Dataset 6": pd.DataFrame({"time": range(250), "value": [x * 3 for x in range(250)]}),
}

# Static Parameters
static_params = pd.DataFrame({
    "Parameter": ["Param1", "Param2", "Param3"],
    "Value": [10, 20, 30],
})

app_ui = ui.page_fluid(
    ui.h1("A Study of Accelerated Gradient Descent for Classification Logistic Regression"),
    ui.div(
        ui.strong("Information Box"),
        ui.p("Use the controls on the left to select datasets and adjust plot limits. The reactive results table is now on the right."),
        class_="alert alert-info"
    ),
    ui.row(
        ui.column(
            3,  # Left column
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
            6,  # Center column
            ui.output_plot("combined_plot")
        ),
        ui.column(
            3,  # Right column
            ui.h2("Reactive Results Table"),
            ui.output_table("results_table")
        )
    )
)

def server(input, output, session):
    # Reactive expression to check validity of ranges
    @reactive.Calc
    def valid_ranges():
        return (input.xmin() < input.xmax()) and (input.ymin() < input.ymax())

    # Show a warning notification whenever the ranges become invalid
    @reactive.Effect
    def show_warning_if_invalid():
        if not valid_ranges():
            ui.notification_show(
                "Invalid axis range provided. Reverting to default (X: 0–250, Y: 0–110).",
                type="error",
                duration=4
            )

    @output
    @render.plot
    def combined_plot():
        selected = input.datasets()

        if valid_ranges():
            x_min, x_max = input.xmin(), input.xmax()
            y_min, y_max = input.ymin(), input.ymax()
        else:
            x_min, x_max = 0, 250
            y_min, y_max = 0, 110

        plt.figure(figsize=(10, 6))
        for name, df in datasets.items():
            if name in selected:
                plt.plot(df["time"], df["value"], label=name)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy [%]")
        plt.title("")
        plt.legend()
        plt.tight_layout()

    @output
    @render.table
    def static_table():
        return static_params

    @output
    @render.table
    def results_table():
        selected = input.datasets()
        results = []
        for name, df in datasets.items():
            results.append({
                "Dataset": name,
                "Min Time": df["time"].min(),
                "Max Time": df["time"].max(),
                "Mean Value": df["value"].mean()
            })
        return pd.DataFrame(results)

app = App(app_ui, server)

