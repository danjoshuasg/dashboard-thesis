import os
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load dataset
df = pd.read_csv('datos.csv')

# Create Dash application
app = Dash(__name__)
app.title = "Dan Santivañez Master Model Experiments"

# Define color schemes for dark and light modes
dark_colors = {
    'background': '#2F3136',
    'text': '#FFFFFF',
    'primary': '#4C72B0',
    'secondary': '#DD8452',
    'tertiary': '#55A868',
    'quaternary': '#C44E52',
    'quinary': '#8172B2',
    'senary': '#DA8BC3',
    'header': '#1E2124'
}

light_colors = {
    'background': '#FFFFFF',
    'text': '#000000',
    'primary': '#4C72B0',
    'secondary': '#DD8452',
    'tertiary': '#55A868',
    'quaternary': '#C44E52',
    'quinary': '#8172B2',
    'senary': '#DA8BC3',
    'header': '#E0E0E0'
}

# Custom CSS
custom_css = '''
    body {
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }
    .header {
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #CCCCCC;
    }
    .dash-dropdown .Select-control {
        background-color: #40444B;
        border-color: #40444B;
    }
    .dash-dropdown .Select-menu-outer {
        background-color: #40444B;
    }
    .dash-dropdown .Select-value-label {
        color: #FFFFFF !important;
    }
    .dash-graph {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    #toggle-button {
        background-color: #4C72B0;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition-duration: 0.4s;
    }
    #toggle-button:hover {
        background-color: #55A868;
        color: white;
    }
        .switch-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }
    .switch-label {
        margin-right: 10px;
        font-size: 14px;
    }
    .main-title {
        text-align: center;
        margin-bottom: 20px;
        font-size: 24px;
        font-weight: bold;
    }
    
'''

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            {custom_css}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div("Thesis: Distillation of Violence Detection in Surveillance Videos Based on MViT Model", 
                 style={'fontStyle': 'italic', 'fontSize': '14px'}),
        html.Div("Author: Dan Santivañez Gutarra", 
                 style={'fontStyle': 'italic', 'fontSize': '14px'}),        
        html.Div([
            html.Span("Dark Mode", className="switch-label"),
            daq.BooleanSwitch(
                id='dark-mode-switch',
                on=True,
                color="#4C72B0"
            )
        ], className="switch-container")
    ], id='header', className='header', style={
        'backgroundColor': '#E0E0E0',
        'padding': '15px 20px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'borderBottom': '2px solid #CCCCCC',
        'lineHeight': '1.2'
    }),

    # Main content
    html.Div([
        html.H1("Multiescale Vision Transformer (MViT) Master Model Experiments", 
        className="main-title", 
        id='main-title'),
        
        # Dropdown for single experiment selection
        html.Div([
            html.Label("Select Experiment for Training and Validation Loss:", id='experiment-selector-single-label'),
            dcc.Dropdown(
                id='experiment-selector-single',
                options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
                value=df['Name Experiment'].unique()[0],
                style={'width': '100%', 'margin': '10px 0'}
            )
        ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),

        # Graph for Training and Validation Loss
        dcc.Graph(id='loss-graph', className='dash-graph'),

        # Dropdown for multiple experiments selection
        html.Div([
            html.Label("Select Experiments for Validation Accuracy Comparison:", id='experiment-selector-multiple-label'),
            dcc.Dropdown(
                id='experiment-selector-multiple',
                options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
                value=[df['Name Experiment'].unique()[0]],
                multi=True,
                style={'width': '100%', 'margin': '10px 0'}
            )
        ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),

        # Graph for Validation Accuracy Comparison
        dcc.Graph(id='accuracy-graph', className='dash-graph'),

        # Dropdowns for metric selection and experiment selection for bar chart
        html.Div([
            html.Label("Select Experiments for Evaluation Metrics:", id='experiment-selector-distribution-label'),
            dcc.Dropdown(
                id='experiment-selector-distribution',
                options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
                value=[df['Name Experiment'].unique()[0]],
                multi=True,
                style={'width': '100%', 'margin': '10px 0'}
            ),
            html.Label("Select Metric for Bar Chart:", id='metric-selector-label'),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'Evaluation Accuracy', 'value': 'Evaluation Accuracy'},
                    {'label': 'Evaluation Precision', 'value': 'Evaluation Precision'},
                    {'label': 'Evaluation F1', 'value': 'Evaluation F1'}
                ],
                value='Evaluation Accuracy',
                style={'width': '100%', 'margin': '10px 0'}
            )
        ], style={'width': '50%', 'margin': 'auto', 'padding': '20px'}),

        # Bar chart for evaluation metrics
        dcc.Graph(id='evaluation-bar-chart', className='dash-graph'),

    ], id='main-content', style={'padding': '20px'})
], id='page-content')

# Update the callback to use the new switch
@app.callback(
    [Output('page-content', 'style'),
     Output('header', 'style'),
     Output('main-content', 'style'),
     Output('experiment-selector-single-label', 'style'),
     Output('experiment-selector-multiple-label', 'style'),
     Output('experiment-selector-distribution-label', 'style'),
     Output('metric-selector-label', 'style'),
     Output('main-title', 'style')],  # Nuevo Output para el estilo del H1
    [Input('dark-mode-switch', 'on')]
)
def toggle_light_dark_mode(dark_mode):
    if dark_mode:
        # Dark mode
        return [
            {'backgroundColor': dark_colors['background']},
            {'backgroundColor': dark_colors['header'], 'color': dark_colors['text'],
             'padding': '15px 20px', 'display': 'flex', 'justifyContent': 'space-between',
             'alignItems': 'center', 'borderBottom': '2px solid #CCCCCC'},
            {'backgroundColor': dark_colors['background'], 'padding': '20px'},
            {'color': dark_colors['text']},
            {'color': dark_colors['text']},
            {'color': dark_colors['text']},
            {'color': dark_colors['text']},
            {'color': dark_colors['text']}  # Nuevo estilo para el H1 en modo oscuro
        ]
    else:
        # Light mode
        return [
            {'backgroundColor': light_colors['background']},
            {'backgroundColor': light_colors['header'], 'color': light_colors['text'],
             'padding': '15px 20px', 'display': 'flex', 'justifyContent': 'space-between',
             'alignItems': 'center', 'borderBottom': '2px solid #CCCCCC'},
            {'backgroundColor': light_colors['background'], 'padding': '20px'},
            {'color': light_colors['text']},
            {'color': light_colors['text']},
            {'color': light_colors['text']},
            {'color': light_colors['text']},
            {'color': light_colors['text']}  # Nuevo estilo para el H1 en modo claro
        ]
# Update other callbacks to use the new switch
@app.callback(
    Output('loss-graph', 'figure'),
    [Input('experiment-selector-single', 'value'),
     Input('dark-mode-switch', 'on')]
)
def update_loss_graph(selected_experiment, dark_mode):
    colors = dark_colors if dark_mode else light_colors
    filtered_df = df[df['Name Experiment'] == selected_experiment]
    
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Training Loss'],
                                  mode='lines+markers', name='Training Loss', line=dict(color=colors['primary'])))
    loss_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Validation Loss'],
                                  mode='lines+markers', name='Validation Loss', line=dict(color=colors['secondary'])))
    
    loss_fig.update_layout(
        title='Training and Validation Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    
    return loss_fig

# Callback for Accuracy Graph
@app.callback(
    Output('accuracy-graph', 'figure'),
    [Input('experiment-selector-multiple', 'value'),
     Input('dark-mode-switch', 'on')]
)
def update_accuracy_graph(selected_experiments, dark_mode):
    colors = dark_colors if dark_mode else light_colors
    accuracy_fig = go.Figure()
    color_list = [colors['primary'], colors['secondary'], colors['tertiary'], colors['quaternary'],
                  colors['quinary'], colors['senary']]

    for i, experiment in enumerate(selected_experiments):
        filtered_df = df[df['Name Experiment'] == experiment]
        accuracy_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Validation Accuracy'],
                                          mode='lines+markers', name=experiment, line=dict(color=color_list[i % len(color_list)])))

    accuracy_fig.update_layout(
        title='Validation Accuracy Comparison over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return accuracy_fig

# Callback for Bar Chart
@app.callback(
    Output('evaluation-bar-chart', 'figure'),
    [Input('experiment-selector-distribution', 'value'),
     Input('metric-selector', 'value'),
     Input('dark-mode-switch', 'on')]
)
def update_bar_chart(selected_experiments, selected_metric, dark_mode):
    colors = dark_colors if dark_mode else light_colors
    bar_fig = go.Figure()

    color_list = [colors['primary'], colors['secondary'], colors['tertiary'], colors['quaternary'],
                  colors['quinary'], colors['senary']]

    y_values = []
    for i, experiment in enumerate(selected_experiments):
        filtered_df = df[df['Name Experiment'] == experiment]
        unique_value = filtered_df[selected_metric].iloc[0]
        bar_fig.add_trace(go.Bar(x=[experiment], y=[unique_value], name=experiment, marker_color=color_list[i % len(color_list)]))
        y_values.append(unique_value)

    if y_values:
        y_min = min(y_values) - 0.01
        y_max = max(y_values) + 0.01
    else:
        y_min, y_max = 0, 1

    bar_fig.update_layout(
        title=f'{selected_metric} for Selected Experiments',
        xaxis_title='Experiment Name',
        yaxis_title=selected_metric,
        yaxis=dict(range=[y_min, y_max]),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return bar_fig

if __name__ == '__main__':
    #app.run_server(debug=True)
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=True)