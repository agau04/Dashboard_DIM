import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import plotly.graph_objects as go
import holidays

st.set_page_config(page_title="Statistiques DIM", layout="wide")

COLOR_PRIMARY = "#507DAE"
COLOR_ALERT = "#BD5153"
BACKGROUND_COLOR = '#F9F9F9'

fr_holidays = holidays.France(years=range(2020, 2031))

# Correction: Rerun sécurisé
if "reload_triggered" not in st.session_state:
    st.session_state.reload_triggered = False

if st.session_state.reload_triggered:
    st.session_state.reload_triggered = False
    st.experimental_rerun()

with st.sidebar:
    if st.button("🔁 Recharger les données"):
        st.cache_data.clear()
        st.session_state.reload_triggered = True

@st.cache_data(ttl=600)
def load_csv_from_url():
    url = "https://sobotram.teliway.com:443/appli/vsobotram/main/extraction.php?sAction=export&idDo=173&sCle=KPI_DIM&sTypeResultat=csv"
    try:
        response = requests.get(url, verify=False, timeout=10)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()
    try:
        content = response.content.decode('utf-8')
    except UnicodeDecodeError:
        content = response.content.decode('iso-8859-1')
    df = pd.read_csv(io.StringIO(content), sep=';', engine='python')
    df.columns = [col.strip() for col in df.columns]
    return df

@st.cache_data(ttl=300)
def preprocess_df(df):
    for col in ['Date_BE', 'Date_depart', 'Date_liv']:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], errors='coerce')
    if 'Souffrance' in df.columns:
        df['Souffrance'] = df['Souffrance'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.strip()
    return df

def jours_ouvres(start, end):
    if pd.isna(start) or pd.isna(end) or end < start:
        return np.nan
    start_np = np.datetime64(start.date())
    end_np = np.datetime64(end.date())
    hol_np = np.array([np.datetime64(d) for d in fr_holidays if start.date() < d <= end.date()], dtype='datetime64[D]')
    jours = np.busday_count(start_np + 1, end_np + 1, holidays=hol_np)
    return max(jours, 1)

@st.cache_data(ttl=300)
def calculate_delta_jours_ouvres(df):
    df['Delta_jours_ouvres'] = df.apply(
        lambda row: jours_ouvres(row['Date_depart_dt'], row['Date_liv_dt']) if pd.notna(row['Date_depart_dt']) and pd.notna(row['Date_liv_dt']) else np.nan,
        axis=1
    )
    return df

@st.cache_data(ttl=300)
def count_souffrance(df):
    if 'Souffrance' not in df.columns:
        return 0, 0
    souffrance_non_null = df['Souffrance'].astype(str).str.strip().replace({'', 'nan', 'NaN', 'None'}, None).dropna()
    return len(souffrance_non_null), len(df)

@st.cache_data(ttl=300)
def extract_departements(df):
    if 'CP' in df.columns:
        df['Departement'] = df['CP'].astype(str).str[:2]
        df = df[df['Departement'].str.match(r'^\d{2}$')]
    else:
        df['Departement'] = None
    return df

def plot_delta_plotly(delta_counts):
    total = delta_counts.sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=delta_counts.index.astype(str),
        y=delta_counts.values,
        text=[f"{int(v)} ({v/total*100:.0f}%)" for v in delta_counts.values],
        textposition='outside',
        marker_color=COLOR_PRIMARY
    ))
    fig.update_layout(
        title="Répartition des délais de livraison",
        xaxis_title="Délai de livraison (jours ouvrés)",
        yaxis_title="Nombre d'expéditions",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white", size=12),
        margin=dict(t=60),
        height=400,
        xaxis=dict(
            showticklabels=True,
            tickfont=dict(color="white"),
            dtick=1
        ),
        yaxis=dict(
            showticklabels=True,
            tickfont=dict(color="white")
        ),
        annotations=[]
    )
    return fig

def plot_souffrance_plotly(count, total):
    labels = ['Avec Souffrance', 'Sans Souffrance']
    values = [count, total - count]
    colors = [COLOR_ALERT, COLOR_PRIMARY]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        textinfo='percent+label',
        marker=dict(colors=colors),
        hole=0.4
    )])
    fig.update_layout(
        title="Proportion des BL avec Souffrance",
        annotations=[],
        margin=dict(t=80),
        height=400
    )
    return fig

def plot_livraison_kpi_plotly(df):
    nb_parties = df['Date_depart_dt'].notna().sum()
    nb_livrees = df['Date_liv_dt'].notna().sum()
    nb_non_livrees = nb_parties - nb_livrees
    labels = ['Livrées', 'Non livrées']
    values = [nb_livrees, nb_non_livrees]
    colors = [COLOR_PRIMARY, COLOR_ALERT]
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='percent+label',
        marker=dict(colors=colors),
        hole=0.3
    )])
    fig.update_layout(
        title="Taux de livraison",
        annotations=[dict(
            text=f"{nb_livrees}/{nb_parties} livrées",
            x=0.5, y=1.15, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14)
        )],
        margin=dict(t=80),
        height=400
    )
    return fig

# --- MAIN ---
st.title("📦 KPI Transport DIM")

df = load_csv_from_url()
if df.empty:
    st.warning("Aucune donnée chargée.")
    st.stop()

df = preprocess_df(df)
df = calculate_delta_jours_ouvres(df)

df_filtered = df.copy()

with st.sidebar:
    st.header("🔍 Filtres")
    if 'Date_BE_dt' in df_filtered:
        min_date = df_filtered['Date_BE_dt'].min().date()
        max_date = df_filtered['Date_BE_dt'].max().date()
        date_range = st.date_input("🗕️ Période Date_BE", value=[min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            df_filtered = df_filtered[
                (df_filtered['Date_BE_dt'] >= pd.to_datetime(date_range[0])) &
                (df_filtered['Date_BE_dt'] <= pd.to_datetime(date_range[1]))
            ]

    if 'Type_Transport' in df_filtered:
        options = df_filtered['Type_Transport'].dropna().unique()
        selected = st.selectbox("🚛 Type Transport", ["(Tous)"] + sorted(options))
        if selected != "(Tous)":
            df_filtered = df_filtered[df_filtered['Type_Transport'] == selected]

    if 'CHRONO' in df_filtered:
        chrono_options = df_filtered['CHRONO'].dropna().unique()
        selected_chrono = st.selectbox("⏱️ CHRONO", ["(Tous)"] + sorted(chrono_options))
        if selected_chrono != "(Tous)":
            df_filtered = df_filtered[df_filtered['CHRONO'] == selected_chrono]


df_filtered = extract_departements(df_filtered)

st.subheader("📋 Données brutes")
df_display = df_filtered.drop(columns=['Date_BE_dt', 'Date_depart_dt', 'Date_liv_dt'], errors='ignore').reset_index(drop=True)
st.dataframe(df_display, use_container_width=True)

col1, col2 = st.columns(2)

if 'Date_liv' in df_filtered:
    df_delta = df_filtered[df_filtered['Date_liv'].notna()]
else:
    df_delta = df_filtered

delta_series = df_delta['Delta_jours_ouvres'].dropna().astype(int)

if not delta_series.empty:
    delta_counts = delta_series.value_counts().sort_index()
    delta_counts = delta_counts[delta_counts.index <= 30]
    with col1:
        st.subheader("📊 Répartition des délais de livraison (jours ouvrés)")
        st.markdown(f"**{delta_counts.sum()} expéditions avec un délai mesuré**")
        st.plotly_chart(plot_delta_plotly(delta_counts), use_container_width=True)
else:
    with col1:
        st.info("Pas de données avec délai mesuré.")

df_souffrance = df_filtered[df_filtered.get('Date_depart', pd.Series([True]*len(df_filtered))).notna()]
souff_count, total_rows = count_souffrance(df_souffrance)
if total_rows > 0:
    with col2:
        st.subheader("⚠️ Analyse Souffrance")
        st.markdown(f"**{souff_count} sur {total_rows} BL avec souffrance**")
        st.plotly_chart(plot_souffrance_plotly(souff_count, total_rows), use_container_width=True)
else:
    with col2:
        st.info("Pas de données analysables pour la souffrance.")

st.subheader("📈 KPI Livraison")
st.plotly_chart(plot_livraison_kpi_plotly(df_filtered), use_container_width=True)

csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button("📅 Export CSV", data=csv, file_name='export_dynamique.csv', mime='text/csv')

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
    df_display.to_excel(writer, sheet_name='Données', index=False)

st.download_button("📅 Export Excel", data=excel_buf.getvalue(), file_name='export_dynamique.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
