from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

import subprocess
import plotly.graph_objects as go


import pickle

from pathlib import Path

# =========================================================
# RUNTIME CONFIGURATION
# =========================================================
# These are populated at startup from command-line arguments in the
# __main__ block below (see `python dashboard.py --help`).

DATA_DIR = None           # set from --data_dir
PREPROCESSED_ROOT = None  # set from --preprocessed_root
START_DATE = None         # set from --start_date
END_DATE = None           # set from --end_date
OUTPUT_PATH_1 = None      # derived from --src_dir
OUTPUT_PATH_2 = None      # derived from --src_dir

"""
    Extract the inner algorithm metrics dictionary from a
    serialized optimization output.

    Different algorithms serialize results under different keys
    (e.g. 'dsa', 'dsa_eb', 'osco', 'strategic').

    Returns:
        dict: dashboard-ready metrics dictionary
    """
def extract_dashboard_data(obj):

    if "dsa_eb" in obj:
        return obj["dsa_eb"]

    if "dsa" in obj:
        return obj["dsa"]

    if "osco" in obj:
        return obj["osco"]

    if "strategic" in obj:
        return obj["strategic"]

    # single-key wrapper
    if isinstance(obj, dict) and len(obj) == 1:
        return list(obj.values())[0]

    raise ValueError(f"Unsupported pickle format: {obj.keys()}")

def make_input_card(title, children, color):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title", style={"fontSize": "22px"}),
            *children
        ]),
        className="mb-3 shadow-sm",
        style={
            "backgroundColor": color 
        }
    )


import base64
import io

def parse_contents(contents):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return pickle.load(io.BytesIO(decoded))
    except Exception as e:
        print("Upload error:", e)
        return None

"""
Run the DSA optimization pipeline or load uploaded results.

Modes:
- "run": execute the optimization subprocess and write results to disk
- "upload": parse uploaded pickle files from the dashboard UI

The generated output pickle is later converted into dashboard-ready
metrics using `extract_dashboard_data`.
"""
def run_dsa(weight_a, weight_b, weight_c, exec_cl, exec_ea, exec_eu, exec_na, eb_looseness, output_path, mode, contents1, contents2, ebpVal, sbVal, projVal, utilVal):
    if mode == "upload":
        if contents1 is None or contents2 is None:
            raise PreventUpdate

        data1 = parse_contents(contents1)
        data2 = parse_contents(contents2)

        if data1 is None or data2 is None:
            fig = go.Figure()
            fig.update_layout(title="Error loading uploaded files")
            return fig

        data1 = data1["dsa_eb"]
        data2 = data2["dsa_eb"]

    elif mode == "run":
        cmd = [
            "python", "run_dsa_eb_dashboard.py",
            "--data_dir", str(DATA_DIR),
            "--output_path", str(output_path),
            "--preprocessed_root", str(PREPROCESSED_ROOT),
            "--start_date", str(START_DATE),
            "--end_date", str(END_DATE),
            "--w_sb", str(sbVal),
            "--w_proj", str(projVal),
            "--w_util", str(utilVal),
            "--w_ebp", str(ebpVal),
            "--eb_ramp_exponent", str(eb_looseness),
            "--override_cycle_grade_score",
            "--cycle_grade_score_A", str(weight_a),
            "--cycle_grade_score_B", str(weight_b),
            "--cycle_grade_score_C", str(weight_c),
        ]

        print("Running DSA subprocess...")
        subprocess.run(cmd, check=True)

    path = output_path
    # print("FILE MODIFIED TIME:", time.ctime(os.path.getmtime(path)))    
    print("Finished DSA subprocess.")

"""
Create grouped bar chart comparing:
- project completion percentages
- stretch/buffer completion
- schedulable utilization

between two dashboard runs.
"""
def create_proj_util_graph(data1, data2):
    df = pd.DataFrame({
        "Metric" : [
            "Proj A","Proj B","Proj C",
            "SB A","SB B","SB C",
            "Utilization"
        ],
        "Weights 1": [
            data1["completion_pct_proj_A"],
            data1["completion_pct_proj_B"],
            data1["completion_pct_proj_C"],
            data1.get("completion_pct_sb_A", 0),
            data1.get("completion_pct_sb_B", 0),
            data1.get("completion_pct_sb_C", 0),
            data1["schedulable_utilization"],
        ],
        "Weights 2": [
            data2["completion_pct_proj_A"],
            data2["completion_pct_proj_B"],
            data2["completion_pct_proj_C"],
            data2.get("completion_pct_sb_A", 0),
            data2.get("completion_pct_sb_B", 0),
            data2.get("completion_pct_sb_C", 0),
            data2["schedulable_utilization"],
        ]
    })

    df = df.melt(id_vars="Metric", var_name="Run", value_name="Value")

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Run",
        barmode="group",
        # title="Project Completion + Utilization",
        color_discrete_map={
            "Weights 1": "#ed5647",
            "Weights 2": "#73a5eb",
        }
    )

    fig.update_layout(
        yaxis_title="Percent",
        template="plotly_white",
        showlegend=False,

        yaxis=dict(
            range=[0, 120],      # max height
            dtick=10,            # grid line every 10
            showgrid=True,
            gridwidth=1
        )
    )

    return fig

"""
    Compare executive allocation deviation from target percentages.

    Positive values indicate over-allocation relative to target.
    Negative values indicate under-allocation.
    """
def create_exec_balance_graph(data1, data2,
                              exec_cl, exec_ea, exec_eu, exec_na,
                              exec_cl2, exec_ea2, exec_eu2, exec_na2):

    targets1 = {
        "CL": exec_cl * 100,
        "EA": exec_ea * 100,
        "EU": exec_eu * 100,
        "NA": exec_na * 100
    }

    targets2 = {
        "CL": exec_cl2 * 100,
        "EA": exec_ea2 * 100,
        "EU": exec_eu2 * 100,
        "NA": exec_na2 * 100
    }

    regions = ["CL", "EA", "EU", "NA"]

    label_map = {
        "CL": "Chile",
        "EA": "East Asia",
        "EU": "Europe",
        "NA": "North America"
    }

    run1_vals = []
    run2_vals = []

    for r in regions:
        actual1 = data1["exec_time_fractions_real"][r] * 100
        target1 = targets1[r]

        actual2 = data2["exec_time_fractions_real"][r] * 100
        target2 = targets2[r]

        diff1 = ((actual1 - target1) / target1) * 100 if target1 != 0 else 0
        diff2 = ((actual2 - target2) / target2) * 100 if target2 != 0 else 0

        run1_vals.append(diff1)
        run2_vals.append(diff2)

    df = pd.DataFrame({
        "Region": [label_map[r] for r in regions] * 2,
        "Run": ["Weights 1"] * 4 + ["Weights 2"] * 4,
        "Difference": run1_vals + run2_vals
    })

    fig = px.bar(
        df,
        x="Region",
        y="Difference",
        color="Run",
        barmode="group",
        # title="Executive Balance (% Off Target)",
        color_discrete_map={
            "Weights 1": "#ed5647",
            "Weights 2": "#73a5eb"
        }
    )

    fig.add_hline(y=0, line_color="red", line_width=2)

    fig.update_layout(
        yaxis_title="% Difference from Target",
        xaxis_title="Executive",
        template="plotly_white",
        showlegend=False,
        height=600
    )

    return fig

def create_single_graph(data, color, label):

    df = pd.DataFrame({
        "Metric": [
            "Proj A", "Proj B", "Proj C",
            "SB A", "SB B", "SB C",
            "Utilization"
        ],
        "Value": [
            data["completion_pct_proj_A"],
            data["completion_pct_proj_B"],
            data["completion_pct_proj_C"],
            data["completion_pct_sb_A"],
            data["completion_pct_sb_B"],
            data["completion_pct_sb_C"],
            data["schedulable_utilization"]
        ]
    })

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        title=label,
        text_auto=".1f"
    )

    fig.update_traces(marker_color=color)

    fig.update_layout(
        yaxis_title="Percent",
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[0, 120], dtick=10)
    )

    return fig

def create_single_eb_graph(
    data,
    exec_cl, exec_ea, exec_eu, exec_na,
    color,
    label
):

    targets = {
        "CL": exec_cl * 100,
        "EA": exec_ea * 100,
        "EU": exec_eu * 100,
        "NA": exec_na * 100
    }

    regions = ["CL", "EA", "EU", "NA"]
    vals = []

    for r in regions:
        actual = data["exec_time_fractions_real"][r] * 100
        target = targets[r]

        diff = ((actual - target) / target) * 100 if target != 0 else 0
        vals.append(diff)

    df = pd.DataFrame({
        "Region": regions,
        "Difference": vals
    })

    fig = px.bar(
        df,
        x="Region",
        y="Difference",
        title=label,
        text_auto=".1f"
    )

    fig.update_traces(marker_color=color)

    fig.add_hline(y=0, line_color="red", line_width=2)

    fig.update_layout(
        yaxis_title="% Difference from Target",
        template="plotly_white",
        showlegend=False,
        height=600
    )

    return fig

def render_page(completion_fig_plotly, idle_time_plotly, eb_fig_plotly):
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
        "display": "inline-block",
        "width": "100%"
    }
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "right": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "display": "inline-block"
    }
    
    


    

    
    
    # app.layout = html.Div([
    #     sidebar,
    #     dbc.Row([
    #         dbc.Col(dcc.Loading(output_graph), width=12),
    #     ])
    # ])
    
    



app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>DSA Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .upload-box {
                width: 100%;
                height: 60px;
                line-height: 60px;
                border: 2px dashed #6c757d;
                border-radius: 10px;
                text-align: center;
                transition: all 0.2s ease-in-out;
                background-color: #f8f9fa;
            }

            .upload-box:hover {
                border-color: #0d6efd;
                background-color: #eef5ff;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
app.config.suppress_callback_exceptions = True

## completion
output_graph = dcc.Graph(id="graph_completion", figure={})

## A
user_input = dbc.Input(id='input_a', type='number', value=10)

## B
user_input_b = dbc.Input(id='input_b', type='number', value=4)

## C
user_input_c = dbc.Input(id='input_c', type='number', value=-100)


## EXEC BALANCE
user_input_cl = dbc.Input(id='input_cl', type='number', value=10)
user_input_ea = dbc.Input(id='input_ea', type='number', value=22.5)
user_input_eu = dbc.Input(id='input_eu', type='number', value=33.75)
user_input_na = dbc.Input(id='input_na', type='number', value=33.75)
output_text_cl = dcc.Markdown(id="text_cl", children='Executive Balance for CL')
output_text_ea = dcc.Markdown(id="text_ea", children='Executive Balance for EA')
output_text_eu = dcc.Markdown(id="text_eu", children='Executive Balance for EU')
output_text_na = dcc.Markdown(id="text_na", children='Executive Balance for NA')

# button
run_button = dbc.Button(
    "Run",
    id="run_button",
    color="primary",
    size="lg",
    className="px-5"
)


## SECOND WEIGHT SET
user_input_a_2 = dbc.Input(id='input_a_2', type='number', value=10)
user_input_b_2 = dbc.Input(id='input_b_2', type='number', value=4)
user_input_c_2 = dbc.Input(id='input_c_2', type='number', value=-100)
user_input_cl_2 = dbc.Input(id='input_cl_2', type='number', value=10)
user_input_ea_2 = dbc.Input(id='input_ea_2', type='number', value=22.5)
user_input_eu_2 = dbc.Input(id='input_eu_2', type='number', value=33.75)
user_input_na_2 = dbc.Input(id='input_na_2', type='number', value=33.75)

## eb_looseness
user_input_eb_looseness = dbc.Input(id='input_eb_looseness', type='number', value=10)
user_input_eb_looseness_2 = dbc.Input(id='input_eb_looseness_2', type='number', value=10)



app.layout = html.Div(

    dbc.Container([
    dbc.Row([

        # LEFT
        dbc.Col([

            html.Label("Algorithm"),
            dcc.Dropdown(
                id="algo_selector_1",
                options=[
                    {"label": "Current", "value": "dsa"},
                    {"label": "Current + Penalty", "value": "dsa_eb"},
                ],
                value="dsa_eb",
                clearable=False,
                className="mb-3"
            ),

            make_input_card("Weights Set 1", [
                html.Label("Project A"), user_input,
                html.Label("Project B"), user_input_b,
                html.Label("Project C"), user_input_c,
            ], "#ed5647"),

            make_input_card("Executive Balance", [
                html.Label("Chile"), user_input_cl,
                html.Label("East Asia"), user_input_ea,
                html.Label("Europe"), user_input_eu,
                html.Label("North America"), user_input_na,
                html.Label("Executive Balance Looseness"), user_input_eb_looseness,
            ], "#ed5647"),
        
            html.Label("Preloaded"),
            dcc.Dropdown(
                id="display_selector_1",
                options=[
                     {"label": "Use Algorithm Output", "value": "algorithm"},
                    {"label": "Upload Pickle File", "value": "upload"},
                ],
                value="algorithm",
                clearable=False
            ),

        ], width=2),

        # CENTER
        dbc.Col([

            dbc.Row([

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Projects", className="mb-2"),
                            dcc.Loading(
                                dcc.Graph(
                                    id="graph_completion",
                                    style={"height": "70vh"}
                                )
                            )
                        ]),
                        className="shadow-sm"
                    ),
                    width=7
                ),

                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5("Executive Balance", className="mb-2"),
                            dcc.Loading(
                                dcc.Graph(
                                    id="graph_eb",
                                    style={"height": "70vh"}
                                )
                            )
                        ]),
                        className="shadow-sm"
                    ),
                    width=5
                ),

            ])

        ], width=8),

            # RIGHT
            dbc.Col([

                html.Label("Algorithm"),
                dcc.Dropdown(
                    id="algo_selector_2",
                    options=[
                        {"label": "Current", "value": "dsa"},
                        {"label": "Current + Penalty", "value": "dsa_eb"},
                    ],
                    value="dsa_eb",
                    clearable=False,
                    className="mb-3"
                ),

                make_input_card("Weights Set 2", [
                    html.Label("Project A"), user_input_a_2,
                    html.Label("Project B"), user_input_b_2,
                    html.Label("Project C"), user_input_c_2,
                ], "#73a5eb"),

                make_input_card("Executive Balance", [
                    html.Label("Chile"), user_input_cl_2,
                    html.Label("East Asia"), user_input_ea_2,
                    html.Label("Eirope"), user_input_eu_2,
                    html.Label("North America"), user_input_na_2,
                    html.Label("Executive Balance Looseness"), user_input_eb_looseness_2,
                ], "#73a5eb"),
            
                html.Label("Preloaded"),
                dcc.Dropdown(
                    id="display_selector_2",
                    options=[
                        {"label": "Use Algorithm Output", "value": "algorithm"},
                        {"label": "Upload Pickle File", "value": "upload"},
                    ],
                    value="algorithm",
                    clearable=False
                ),

            ], width=2),

        ], align="start"),

    dbc.Row([
    dbc.Col(
        dbc.Card(
            dbc.CardBody([

                
                html.H5("Display Mode", className="text-center mt-4 mb-2"),

                dbc.RadioItems(
                    id="compare_mode",
                    options=[
                        {"label": "Compare Two Runs", "value": "both"},
                        {"label": "Only Weights Set 1", "value": "left"},
                        {"label": "Only Weights Set 2", "value": "right"},
                    ],
                    value="both",
                    inline=False,
                    className="d-flex flex-column align-items-center"
                ),

                
                html.H5("Upload Files", className="text-center mb-3"),

                html.Div(
                    id="upload_container_1",
                    children=[
                        dcc.Upload(
                            id='upload_data_1',
                            children=html.Div([
                                'Drag & Drop or ',
                                html.A('Select File for Weights 1')
                            ]),
                            className="upload-box mb-3"
                        ),
                    ]
                ),

                html.Div(
                    id="upload_container_2",
                    children=[
                        dcc.Upload(
                            id='upload_data_2',
                            children=html.Div([
                                'Drag & Drop or ',
                                html.A('Select File for Weights 2')
                            ]),
                            className="upload-box"
                        ),
                    ]
                ),

                # BUTTON
                html.Div(
                    run_button,
                    className="d-flex justify-content-center mt-4"
                )

            ]),
            className="shadow-lg p-3"
        ),
        width=6
    )
], justify="center")
], fluid=True),

style={
    "fontSize": "18px",
    "paddingTop": "20px",
    "paddingLeft": "20px",
    "paddingRight": "20px",
    "paddingBottom": "20px",
}
)

## maybe this has to go after the if statement. like layout created first
@app.callback(
    [Output("graph_completion", "figure"),
    Output("graph_eb", "figure"),],

    [Input("run_button", "n_clicks"),],
    [
        State('input_a', 'value'),
        State('input_b', 'value'),
        State('input_c', 'value'),
        State('input_cl', 'value'),
        State('input_ea', 'value'),
        State('input_eu', 'value'),
        State('input_na', 'value'),
        State('input_a_2', 'value'),
        State('input_b_2', 'value'),
        State('input_c_2', 'value'),
        State('input_cl_2', 'value'),
        State('input_ea_2', 'value'),
        State('input_eu_2', 'value'),
        State('input_na_2', 'value'),
        State('input_eb_looseness', 'value'),
        State('input_eb_looseness_2', 'value'),
        # State("mode_selector", "value"),
        State("upload_data_1", "contents"),
        State("upload_data_2", "contents"),
        State("algo_selector_1", "value"),
        State("display_selector_1", "value"),
        State("algo_selector_2", "value"),
        State("display_selector_2", "value"),
        State("compare_mode", "value"),
    ]
)

def cb_render(n_clicks, num_a, num_b, num_c, 
              exec_cl, exec_ea, exec_eu, exec_na, 
              num_a2, num_b2, num_c2, exec_cl2, exec_ea2, exec_eu2, exec_na2,
              eb_looseness, eb_looseness2,
            #   mode,
              contents1, contents2,
              algorithm1, display1,
              algorithm2, display2,
              compare_mode):
    if n_clicks is None:
        raise PreventUpdate
    run_left = compare_mode in ["both", "left"]
    run_right = compare_mode in ["both", "right"]

    print("CLICKS:", n_clicks)
    # defaults
    num_a = 10 if num_a is None else num_a
    num_b = 4 if num_b is None else num_b
    num_c = -100 if num_c is None else num_c

    print("RUNNING DSA WITH:", num_a, num_b, num_c)
    print("RUN 2:", num_a2, num_b2, num_c2)

    # SECOND RUN
    num_a2 = 0.25 if num_a2 is None else num_a2
    num_b2 = 0.25 if num_b2 is None else num_b2
    num_c2 = 0.25 if num_c2 is None else num_c2
    

    # Executive balance targets are entered as percentages in the UI
    # but stored internally as fractions.
    exec_cl = (10 if exec_cl is None else exec_cl) / 100
    exec_ea = (22.5 if exec_ea is None else exec_ea) / 100
    exec_eu = (33.75 if exec_eu is None else exec_eu) / 100
    exec_na = (33.75 if exec_na is None else exec_na) / 100

    exec_cl2 = (10 if exec_cl2 is None else exec_cl2) / 100
    exec_ea2 = (22.5 if exec_ea2 is None else exec_ea2) / 100
    exec_eu2 = (33.75 if exec_eu2 is None else exec_eu2) / 100
    exec_na2 = (33.75 if exec_na2 is None else exec_na2) / 100

    eb_looseness = 10 if eb_looseness is None else eb_looseness
    eb_looseness2 = 10 if eb_looseness2 is None else eb_looseness2

    print("RAW VALUES:")
    print("num_a:", num_a)
    print("num_b:", num_b)
    print("num_c:", num_c)
    print("exec_cl:", exec_cl)
    print("exec_ea:", exec_ea)
    print("exec_eu:", exec_eu)
    print("exec_na:", exec_na)

    # if not (0 <= num_a <= 1 and 0 <= num_b <= 1 and 0 <= num_c <= 1):
    #     fig = go.Figure()
    #     fig.update_layout(title="Weights must be between 0 and 1")
    #     return fig
    data1 = None
    data2 = None

    # LEFT SIDE
    if run_left:

        # UPLOADED PICKLE
        if display1 == "upload":

            if contents1 is None:
                raise PreventUpdate

            parsed = parse_contents(contents1)

            if parsed is None:
                fig = go.Figure()
                fig.update_layout(title="Error loading uploaded file")
                return fig, fig

            data1 = extract_dashboard_data(parsed)

        # USE ALGORITHM OUTPUT
        elif display1 == "algorithm":

            # ONLY DSA RUNS LIVE
            if algorithm1 == "dsa":

                run_dsa(
                    num_a, num_b, num_c,
                    exec_cl, exec_ea, exec_eu, exec_na,
                    eb_looseness,
                    OUTPUT_PATH_1,
                    "run",
                    None, None,
                    0, 0.25, 0.25, 0.50
                )

                with open(OUTPUT_PATH_1, "rb") as f:
                    data1 = extract_dashboard_data(pickle.load(f))

                # print("DATA1:", data1)

            # EVERYTHING ELSE USES PRELOADED OUTPUTS
            elif algorithm1 == "dsa_eb":
                run_dsa(
                    num_a, num_b, num_c,
                    exec_cl, exec_ea, exec_eu, exec_na,
                    eb_looseness,
                    OUTPUT_PATH_1,
                    "run",
                    None, None,
                    0.995, 0.002, 0.002, 0.001
                )

                with open(OUTPUT_PATH_1, "rb") as f:
                    data1 = extract_dashboard_data(pickle.load(f))


        # RIGHT SIDE
    if run_right:

        # UPLOADED PICKLE
        if display2 == "upload":

            if contents2 is None:
                raise PreventUpdate

            parsed = parse_contents(contents2)

            if parsed is None:
                fig = go.Figure()
                fig.update_layout(title="Error loading uploaded file")
                return fig, fig

            data2 = extract_dashboard_data(parsed)

        # USE ALGORITHM OUTPUT
        elif display2 == "algorithm":

            # ONLY DSA RUNS LIVE
            if algorithm2 == "dsa":

                run_dsa(
                    num_a2, num_b2, num_c2,
                    exec_cl2, exec_ea2, exec_eu2, exec_na2,
                    eb_looseness2,
                    OUTPUT_PATH_2,
                    "run",
                    None, None,
                    0, 0.25, 0.25, 0.50
                )

                with open(OUTPUT_PATH_2, "rb") as f:
                    data2 = extract_dashboard_data(pickle.load(f))

                # print("DATA1:", data1)

            # EVERYTHING ELSE USES PRELOADED OUTPUTS
            elif algorithm2 == "dsa_eb":
                run_dsa(
                    num_a2, num_b2, num_c2,
                    exec_cl2, exec_ea2, exec_eu2, exec_na2,
                    eb_looseness2,
                    OUTPUT_PATH_2,
                    "run",
                    None, None,
                    0.995, 0.002, 0.002, 0.001
                )

                with open(OUTPUT_PATH_2, "rb") as f:
                    data2 = extract_dashboard_data(pickle.load(f))


    # print("DATA1:", type(data1), data1 is None)
    # print("DATA2:", type(data2), data2 is None)
    if compare_mode == "left":
        proj_fig = create_single_graph(data1, "#ed5647", "Weights 1")

    elif compare_mode == "right":
        proj_fig = create_single_graph(data2, "#73a5eb", "Weights 2")

    else:
        proj_fig = create_proj_util_graph(data1, data2)


        # EB GRAPH
    if compare_mode == "left":
        eb_fig = create_single_eb_graph(
            data1, exec_cl, exec_ea, exec_eu, exec_na,
            "#ed5647", "Weights 1"
        )

    elif compare_mode == "right":
        eb_fig = create_single_eb_graph(
            data2, exec_cl2, exec_ea2, exec_eu2, exec_na2,
            "#73a5eb", "Weights 2"
        )

    else:
        eb_fig = create_exec_balance_graph(
            data1, data2,
            exec_cl, exec_ea, exec_eu, exec_na,
            exec_cl2, exec_ea2, exec_eu2, exec_na2
        )

    return [proj_fig, eb_fig]

@app.callback(
    Output("upload_data_1", "style"),
    Input("upload_data_1", "filename")
)
def highlight_upload_1(filename):
    base_style = {
        'width': '100%',
        'height': '50px',
        'lineHeight': '50px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'marginBottom': '10px'
    }
    if filename:
        base_style["borderColor"] = "green"
        base_style["backgroundColor"] = "#e6ffe6"
    return base_style

@app.callback(
    Output("upload_data_2", "style"),
    Input("upload_data_2", "filename")
)
def highlight_upload_2(filename):
    base_style = {
        'width': '100%',
        'height': '50px',
        'lineHeight': '50px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'marginBottom': '10px'
    }
    if filename:
        base_style["borderColor"] = "green"
        base_style["backgroundColor"] = "#e6ffe6"
    return base_style

# @app.callback(
#     Output("upload_container", "style"),
#     Input("display_selector_1", "value"),
#     Input("display_selector_2", "value"),
# )
# def render_mode_inputs(display1, display2):

#     if display1 == "upload" or display2 == "upload":
#         return {"display": "block"}

#     return {"display": "none"}

@app.callback(
    Output("upload_container_1", "style"),
    Input("display_selector_1", "value"),
)
def toggle_upload_1(display1):

    if display1 == "upload":
        return {"display": "block"}

    return {"display": "none"}

@app.callback(
    Output("upload_container_2", "style"),
    Input("display_selector_2", "value"),
)
def toggle_upload_2(display2):

    if display2 == "upload":
        return {"display": "block"}

    return {"display": "none"}

    
def toggle_upload_visibility(mode):
    if mode == "upload":
        return {"display": "block"}
    else:
        return {"display": "none"}
  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PULSAR interactive dashboard")
    parser.add_argument(
        "--src_dir", required=True,
        help="Path to Codes/short_term/src; live-run output pickles are written to dsa_eb_dash/ inside it.",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the dataDashboard directory (same as --data_dir for run_dsa_eb_dashboard.py).",
    )
    parser.add_argument(
        "--preprocessed_root", required=True,
        help="Path to the preprocessed root (contains year_YYYY/realized_weather.pkl).",
    )
    parser.add_argument(
        "--start_date", required=True,
        help="Cycle start date (YYYY-MM-DD), e.g. 2017-10-01.",
    )
    parser.add_argument(
        "--end_date", required=True,
        help="Cycle end date (YYYY-MM-DD), e.g. 2018-09-30.",
    )
    parser.add_argument(
        "--port", type=int, default=8051,
        help="Port for the Dash server (default: 8051).",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    DATA_DIR = Path(args.data_dir)
    PREPROCESSED_ROOT = Path(args.preprocessed_root)
    START_DATE = args.start_date
    END_DATE = args.end_date
    OUTPUT_PATH_1 = src_dir / "dsa_eb_dash/dsa_eb_dashboard_inputs1.pkl"
    OUTPUT_PATH_2 = src_dir / "dsa_eb_dash/dsa_eb_dashboard_inputs2.pkl"

    app.run(debug=True, port=args.port)