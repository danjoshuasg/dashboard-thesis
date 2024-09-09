import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Cargar el dataset desde un archivo CSV
# Asegúrate de reemplazar 'tu_archivo.csv' con la ruta de tu archivo CSV
df = pd.read_csv('datos.csv')

# Crear la aplicación Dash
app = Dash(__name__)
app.title = "Experiment Dashboard"

# Estilo global con fuente similar a Poppins
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins', sans-serif;
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

# Layout de la aplicación
app.layout = html.Div([
    html.H1("Experiment Dashboard", style={'textAlign': 'center'}),

    # Dropdown para seleccionar un solo experimento (affects Training and Validation Loss)
    html.Div([
        html.Label("Select Experiment for Training and Validation Loss:"),
        dcc.Dropdown(
            id='experiment-selector-single',
            options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
            value=df['Name Experiment'].unique()[0],  # Valor inicial
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'padding': '20px'}),

    # Gráfico de líneas para Training Loss y Validation Loss
    dcc.Graph(id='loss-graph'),

    # Dropdown para seleccionar múltiples experimentos (affects Validation Accuracy)
    html.Div([
        html.Label("Select Experiments for Validation Accuracy Comparison:"),
        dcc.Dropdown(
            id='experiment-selector-multiple',
            options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
            value=[df['Name Experiment'].unique()[0]],  # Valor inicial
            multi=True,
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'padding': '20px'}),

    # Gráfico de líneas para Validation Accuracy
    dcc.Graph(id='accuracy-graph'),

    # Dropdowns para seleccionar métricas y experimentos para distribuciones
    html.Div([
        html.Label("Select Experiments for Evaluation Metrics:"),
        dcc.Dropdown(
            id='experiment-selector-distribution',
            options=[{'label': name, 'value': name} for name in df['Name Experiment'].unique()],
            value=[df['Name Experiment'].unique()[0]],  # Valor inicial
            multi=True,
            style={'width': '50%', 'margin': 'auto'}
        ),
        html.Label("Select Metric for Bar Chart:"),
        dcc.Dropdown(
            id='metric-selector',
            options=[
                {'label': 'Evaluation Accuracy', 'value': 'Evaluation Accuracy'},
                {'label': 'Evaluation Precision', 'value': 'Evaluation Precision'},
                {'label': 'Evaluation F1', 'value': 'Evaluation F1'}
            ],
            value='Evaluation Accuracy',  # Valor inicial
            style={'width': '50%', 'margin': 'auto'}
        )
    ], style={'padding': '20px'}),

    # Gráfico de barras para distribuciones
    dcc.Graph(id='evaluation-bar-chart'),
])

# Callback para actualizar los gráficos de Training y Validation Loss basados en la selección única del experimento
@app.callback(
    Output('loss-graph', 'figure'),
    [Input('experiment-selector-single', 'value')]
)
def update_loss_graph(selected_experiment):
    filtered_df = df[df['Name Experiment'] == selected_experiment]

    # Gráfico de líneas para Training Loss y Validation Loss
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Training Loss'],
                                  mode='lines+markers', name='Training Loss', line=dict(dash='dash')))
    loss_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Validation Loss'],
                                  mode='lines+markers', name='Validation Loss', line=dict(dash='dot')))
    loss_fig.update_layout(title='Training and Validation Loss over Epochs', xaxis_title='Epoch', yaxis_title='Loss')

    return loss_fig

# Callback para actualizar el gráfico de Validation Accuracy basado en la selección múltiple de experimentos
@app.callback(
    Output('accuracy-graph', 'figure'),
    [Input('experiment-selector-multiple', 'value')]
)
def update_accuracy_graph(selected_experiments):
    accuracy_fig = go.Figure()

    for experiment in selected_experiments:
        filtered_df = df[df['Name Experiment'] == experiment]
        accuracy_fig.add_trace(go.Scatter(x=filtered_df['Epoch'], y=filtered_df['Validation Accuracy'],
                                          mode='lines+markers', name=experiment))

    accuracy_fig.update_layout(title='Validation Accuracy Comparison over Epochs', xaxis_title='Epoch', yaxis_title='Accuracy')

    return accuracy_fig

# Callback para actualizar los gráficos de barras de distribución basados en la selección múltiple de experimentos y métrica
@app.callback(
    Output('evaluation-bar-chart', 'figure'),
    [Input('experiment-selector-distribution', 'value'),
     Input('metric-selector', 'value')]
)
def update_bar_chart(selected_experiments, selected_metric):
    # Crear gráfico de barras
    bar_fig = go.Figure()

    # Para cada experimento seleccionado, obtener el valor único de la métrica seleccionada y añadirlo al gráfico
    y_values = []
    for experiment in selected_experiments:
        filtered_df = df[df['Name Experiment'] == experiment]
        unique_value = filtered_df[selected_metric].iloc[0]  # Obtener el valor único (suponiendo que solo hay uno)
        bar_fig.add_trace(go.Bar(x=[experiment], y=[unique_value], name=experiment))
        y_values.append(unique_value)

    # Calcular el rango dinámico del eje Y para evidenciar las diferencias pequeñas
    if y_values:
        y_min = min(y_values) - 0.01  # Un pequeño margen por debajo
        y_max = max(y_values) + 0.01  # Un pequeño margen por encima
    else:
        y_min, y_max = 0, 1  # Rango por defecto si no hay valores

    bar_fig.update_layout(title=f'{selected_metric} for Selected Experiments',
                          xaxis_title='Experiment Name',
                          yaxis_title=selected_metric,
                          yaxis=dict(range=[y_min, y_max]))  # Rango dinámico del eje Y

    return bar_fig

if __name__ == '__main__':
    # Obtener el puerto del entorno proporcionado por Render
    port = int(os.environ.get('PORT', 8050))
    # Ejecutar el servidor Dash
    app.run_server(host='0.0.0.0', port=port, debug=True)