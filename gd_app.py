# INSTALL AND RUN
# pip install shiny pandas matplotlib
# shiny run --reload gd_app.py --port 8001
# go to website:
# http://127.0.0.1:8001

from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import ast


df = pd.read_csv("logistic_regression_results2.csv")

# csv turn string to float
def parse_accuracies(acc_str):
    return ast.literal_eval(acc_str)

df["Accuracies"] = df["Accuracies"].apply(parse_accuracies)

methods_list = df["Method"].unique()

static_params = pd.DataFrame({
    "Method": ["GD", "NAG", "PAG", "ADAM", "ADAGrad"],
    "LR": [8e-5,8e-5,1e-5,0.58,1.34],
    "Momentum [1]":["-",0.85,0.8,0.6,"-"],
    "Momentum [2]":["-","-","-",0.999,"-"]
    
})

app_ui = ui.page_fluid(
    ui.h1("A Study of Accelerated Gradient Descent for Classification with Logistic Regression"),
    ui.div(
        ui.strong("README!!!"),
        ui.p("This page describes the results of the project work of group 5 in the FA2024 round of STAT4060J at UM-SJTU Joint Institute. A logistic classification problem was investigated, with different optimization methods based on standard gradient descent. The goal was to get an insight into how gradient descent can accelerated, to solve for accurate classifiers in minimum time. Below is displayed an interactive plot with a run when the hyper parameters were fine-tuned. Different regions can be analyzed via the Axis min/max sliders, and convergence trajectories can be removed and selected using the checkboxes. The dataset 'airline passenger satisfaction', with 24 features after cleaning, available at Kaggle was used, and the logistic regression classifier is trained to predict is satisfied or dissatisfied based on these paramters. All methods were designed and implemented in python."),
        class_="alert alert-info"
    ),
    ui.row(
        ui.column(
            3,
            ui.h3("Plot Control"),
            ui.input_checkbox_group(
                "selected_methods", 
                ui.tags.strong("Select Methods to Display:"),
                {method: method for method in methods_list}, 
                selected=list(methods_list)
            ),
            ui.tags.strong("Plot Axis Control:"),
            ui.input_slider("xmin", "Min (Iterations)", min=0, max=100, value=0),
            ui.input_slider("xmax", "Max (Iterations)", min=0, max=100, value=100),
            ui.input_slider("ymin", "Min (Accuracy)", min=0, max=1.1, value=0.5),
            ui.input_slider("ymax", " Max (Accuracy)", min=0, max=1.1, value=0.9),


        ),
        ui.column(
            5,
            ui.h3("Convergence Plot"),
            ui.output_plot("combined_plot"),
            ui.h3("Discussion"),
            ui.div(
        "The results are clear, that there is no real difference for this problem when using accelerated gradient methods. This can largely be attributed to the fact that the problem is convex. Therefore, with fine tuning, there is no difference. Our initial experiments showed that accelerated gradient methods outperformed basic gradient descent with 'standard' hyper parameters."),
        ),
        ui.column(
            4,
            ui.h3("Hyper Parameters"),
            ui.div(
        "These are the parameters used in training the classifier which led to the displayed plots."),
            ui.output_table("static_table"),
            ui.h3("Convergence Data"),
                        ui.div(
        "Iterations to convergence, time per iteration, and total time until convergence, for easy comparison between selected datasets."),
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
                "Invalid axis range provided. Reverting to defaults.",
                type="error",
                duration=4
            )

    @output
    @render.plot
    def combined_plot():
        if valid_ranges():
            x_min, x_max = input.xmin(), input.xmax()
            y_min, y_max = input.ymin(), input.ymax()
        else:
            x_min, x_max = 0, 100
            y_min, y_max = 0.3, 0.9

        plt.figure(figsize=(10, 6))

        selected = input.selected_methods()

        df_filtered = df[df["Method"].isin(selected)]

        for idx, row in df_filtered.iterrows():
            accuracies = row["Accuracies"]
            actual_iterations = row["Actual Iterations"]

            x_max_clamped = min(x_max, actual_iterations)
            x_values = range(x_min, x_max_clamped)
            y_values = accuracies[x_min:x_max_clamped]

            plt.plot(x_values, y_values, label=row["Method"])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy [0,1]")
        # plt.title("Accuracy Over Iterations by Method")
        plt.legend()
        plt.tight_layout()

    @output
    @render.table
    def results_table():
        selected = input.selected_methods()
        df_results = df[df["Method"].isin(selected)][[
            "Method", 
            "Convergence Iteration", 
            "Convergence Time (s)", 
            "Time/Iteration (ms)"
        ]].copy()

        df_results = df_results.rename(columns={
            "Convergence Iteration": "Conv. Iter.",
            "Convergence Time (s)": "Time (s)",
            "Time/Iteration (ms)": "ms/Iter"
        })

        df_results["Method"] = df_results["Method"].replace({
            "Nestorov Accelerated Gradient": "NAG",
            "Basic Gradient Descent": "GD",
            "Polyak Accelerated Gradient": "PAG"
        })

        df_results["Time (s)"] = df_results["Time (s)"].round(3)
        df_results["ms/Iter"] = df_results["ms/Iter"].round(2)
        df_results["Conv. Iter."] = df_results["Conv. Iter."].astype(int)

        return df_results

    @output
    @render.table
    def static_table():
        return static_params

app = App(app_ui, server)
