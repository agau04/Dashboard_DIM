import pandas as pd
import numpy as np
import io
import requests
from datetime import datetime
import holidays

import dash
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import base64

# --- CONSTANTES CHARTRE GRAPHIQUE ---
BG_COLOR = '#121212'
TEXT_COLOR = '#E0E0E0'
ACCENT_COLOR = '#00B0FF'
ALERT_COLOR = '#FF5252'
CARD_BG_COLOR = '#1E1E1E'
BORDER_RADIUS = '8px'
FONT_FAMILY = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"

# --- DATA SOURCE ---
DATA_URL = "https://example.com/ton_fichier.csv"  # Mets ton URL ou chemin local ici

# --- Initialisation Dash ---
app = dash.Dash(__name__)
server = app.server

# --- Fonction pour charger les données ---
def load_data():
    # Si local: pd.read_csv("chemin.csv")
    # Ici pour ex. je fais avec url:
    try:
        r = requests.get(DATA_URL)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=';')
    except Exception as e:
        print("Erreur chargement données:", e)
        df = pd.DataFrame()
    return df

# --- Préparation et nettoyage ---
def prepare_data(df):
    if df.empty:
        return df

    # Convertir dates
    for col in ['Date_BE', 'Date_depart', 'Date_liv']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Ajouter colonnes utiles, ex. délai livraison
    if 'Date_BE' in df.columns and 'Date_liv' in df.columns:
        df['delai_livraison'] = (df['Date_liv'] - df['Date_BE']).dt.days.clip(lower=0)

    # Ajouter colonne "Souffrance" exemple
    if 'delai_livraison' in df.columns:
        df['Souffrance'] = np.where(df['delai_livraison'] > 3, 'Oui', 'Non')

    # Nettoyer doublons, valeurs manquantes etc selon besoin

    return df

# --- Fonctions graphiques (exemple) ---
def make_delai_bar_chart(df):
    if df.empty or 'delai_livraison' not in df.columns:
        return go.Figure()
    counts = df['delai_livraison'].value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=ACCENT_COLOR,
        text=counts.values,
        textposition='outside',
    ))
    fig.update_layout(
        title="Répartition des délais de livraison (jours)",
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        xaxis=dict(title="Délai (jours)", gridcolor='#333'),
        yaxis=dict(title="Nombre", gridcolor='#333'),
        margin=dict(t=60),
        height=350,
    )
    return fig

def make_souffrance_pie_chart(df):
    if df.empty or 'Souffrance' not in df.columns:
        return go.Figure()
    counts = df['Souffrance'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [ALERT_COLOR if lbl=='Oui' else ACCENT_COLOR for lbl in labels]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
    ))
    fig.update_layout(
        title="Proportion des BL avec Souffrance",
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        paper_bgcolor=BG_COLOR,
        margin=dict(t=60),
        height=350,
    )
    return fig

# --- Layout ---
app.layout = html.Div(style={'backgroundColor': BG_COLOR, 'color': TEXT_COLOR, 'fontFamily': FONT_FAMILY, 'minHeight': '100vh', 'padding': '15px'}, children=[
    html.H1("📦 Dashboard Transport DIM", style={'color': ACCENT_COLOR}),

    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div(style={'flex': '1', 'backgroundColor': CARD_BG_COLOR, 'padding': '20px', 'borderRadius': BORDER_RADIUS}, children=[
            html.H3("Filtres", style={'color': ACCENT_COLOR}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date_placeholder_text='Date début',
                end_date_placeholder_text='Date fin',
                display_format='DD/MM/YYYY',
                style={'backgroundColor': CARD_BG_COLOR, 'color': TEXT_COLOR, 'borderRadius': BORDER_RADIUS}
            ),
            html.Br(), html.Br(),
            dcc.Dropdown(
                id='type-transport-dropdown',
                options=[],
                placeholder="Sélectionner un type de transport",
                clearable=True,
                style={'backgroundColor': CARD_BG_COLOR, 'color': TEXT_COLOR, 'borderRadius': BORDER_RADIUS}
            ),
            html.Br(),
            html.Button("Charger les données", id='load-data-button', style={
                'backgroundColor': ACCENT_COLOR,
                'color': BG_COLOR,
                'border': 'none',
                'padding': '10px',
                'borderRadius': BORDER_RADIUS,
                'cursor': 'pointer',
                'fontWeight': '600',
                'width': '100%',
            }),
            html.Div(id='load-message', style={'marginTop': '10px', 'color': ALERT_COLOR})
        ]),
        html.Div(style={'flex': '3', 'backgroundColor': CARD_BG_COLOR, 'padding': '20px', 'borderRadius': BORDER_RADIUS, 'overflowY': 'auto'}, children=[
            html.H3("Données brutes", style={'color': ACCENT_COLOR}),
            dash_table.DataTable(
                id='data-table',
                columns=[],
                data=[],
                filter_action='native',
                page_size=10,
                style_header={
                    'backgroundColor': ACCENT_COLOR,
                    'color': BG_COLOR,
                    'fontWeight': 'bold',
                    'borderRadius': f'{BORDER_RADIUS} {BORDER_RADIUS} 0 0',
                },
                style_cell={
                    'backgroundColor': CARD_BG_COLOR,
                    'color': TEXT_COLOR,
                    'textAlign': 'left',
                    'fontFamily': FONT_FAMILY,
                    'minWidth': '100px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#292929'}
                ],
                style_table={'overflowX': 'auto'},
            ),
            html.Br(),
            html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    dcc.Graph(id='delai-bar-chart')
                ]),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Graph(id='souffrance-pie-chart')
                ]),
            ])
        ])
    ])
])

# --- Callbacks ---

@app.callback(
    Output('load-message', 'children'),
    Output('data-table', 'columns'),
    Output('data-table', 'data'),
    Output('type-transport-dropdown', 'options'),
    Input('load-data-button', 'n_clicks'),
    State('date-picker-range', 'start_date'),
    State('date-picker-range', 'end_date'),
)
def update_data(n_clicks, start_date, end_date):
    if not n_clicks:
        return "", [], [], []
    df = load_data()
    if df.empty:
        return "Erreur chargement données", [], [], []

    df = prepare_data(df)

    # Filtrer par date_BE si spécifié
    if start_date:
        df = df[df['Date_BE'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date_BE'] <= pd.to_datetime(end_date)]

    # Colonnes tableau
    columns=[{"name": c, "id": c} for c in df.columns]

    # Options type transport
    options = [{"label": t, "value": t} for t in sorted(df['Type_transport'].dropna().unique())] if 'Type_transport' in df.columns else []

    return "", columns, df.to_dict('records'), options

@app.callback(
    Output('data-table', 'data'),
    Input('type-transport-dropdown', 'value'),
    State('data-table', 'data'),
)
def filter_type_transport(selected_type, rows):
    if rows is None or len(rows) == 0:
        return []
    df = pd.DataFrame(rows)
    if selected_type:
        df = df[df['Type_transport'] == selected_type]
    return df.to_dict('records')

@app.callback(
    Output('delai-bar-chart', 'figure'),
    Output('souffrance-pie-chart', 'figure'),
    Input('data-table', 'data'),
)
def update_charts(data):
    df = pd.DataFrame(data)
    fig1 = make_delai_bar_chart(df)
    fig2 = make_souffrance_pie_chart(df)
    return fig1, fig2

# --- Run server ---
if __name__ == '__main__':
    app.run(debug=True)
