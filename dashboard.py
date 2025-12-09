import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import plotly.graph_objects as go
import holidays
from streamlit_datetime_range_picker import datetime_range_picker
import urllib3
import plotly.io as pio

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------------
# CONFIG UI
# --------------------------

st.set_page_config(page_title="Statistiques DIM", layout="wide")

pio.templates["dim_theme"] = pio.templates["plotly_white"]
pio.templates["dim_theme"].layout.font.color = "black"
pio.templates["dim_theme"].layout.xaxis.color = "black"
pio.templates["dim_theme"].layout.yaxis.color = "black"

pio.templates.default = "dim_theme"

st.markdown("""
<style>
header[data-testid="stHeader"] { display: none; }
div[data-testid="stMainBlockContainer"] {
    padding-top: 0.9rem !important;
    margin-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)

COLOR_PRIMARY = "#507DAE"
COLOR_ALERT = "#BD5153"


# ==========================
# CACHES OPTIMISÉS
# ==========================

@st.cache_resource
def get_holiday_array():
    fr_holidays = holidays.France(years=range(2020, 2031))
    return np.array(list(fr_holidays), dtype="datetime64[D]")


@st.cache_data(ttl=600)
def download_csv_raw():
    url = "https://sobotram.teliway.com:443/appli/vsobotram/main/extraction.php?sAction=export&idDo=173&sCle=KPI_DIM&sTypeResultat=csv"
    headers = {"User-Agent": "Streamlit-DIM/1.0"}
    response = requests.get(url, verify=False, timeout=50, stream=True, headers=headers)
    response.raise_for_status()
    return response.raw.read()


@st.cache_data(ttl=600)
def parse_csv(raw_bytes):
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("iso-8859-1")
    df = pd.read_csv(io.StringIO(text), sep=';', quotechar='"', engine='python')
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(ttl=600)
def preprocess_and_compute(df):
    df = df.copy()
    date_cols = [c for c in ['Date_BE', 'Date_depart', 'Date_liv', 'Date_rdv'] if c in df.columns]

    for col in date_cols:
        df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce")

    if 'Souffrance' in df.columns:
        df['Souffrance'] = df['Souffrance'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True).str.strip()

    mask = df['Date_depart_dt'].notna() & df['Date_liv_dt'].notna()
    if mask.any():
        dep_np = df.loc[mask, 'Date_depart_dt'].values.astype("datetime64[D]")
        liv_np = df.loc[mask, 'Date_liv_dt'].values.astype("datetime64[D]")
        holidays_np = get_holiday_array()
        delta = np.busday_count(dep_np + 1, liv_np + 1, holidays=holidays_np)
        df['Delta_jours_ouvres'] = np.nan
        df.loc[mask, 'Delta_jours_ouvres'] = np.maximum(delta, 1)
    else:
        df['Delta_jours_ouvres'] = np.nan

    return df


# ==========================
# CHARGEMENT DES DONNÉES
# ==========================

raw_bytes = download_csv_raw()
df = parse_csv(raw_bytes)
if df.empty:
    st.warning("Aucune donnée chargée.")
    st.stop()
df = preprocess_and_compute(df)


# --------------------------
# SIDEBAR Reload
# --------------------------

if "reload_triggered" not in st.session_state:
    st.session_state.reload_triggered = False

if st.session_state.reload_triggered:
    st.session_state.reload_triggered = False
    st.experimental_rerun()

with st.sidebar:
    if st.button("🔁 Recharger les données"):
        st.cache_data.clear()
        st.session_state.reload_triggered = True


# --------------------------
# KPIs Functions
# --------------------------

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
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=12),
        margin=dict(t=60),
        height=400
    )
    return fig


def count_souffrance(df):
    if 'Souffrance' not in df.columns:
        return 0, 0
    cleaned = df['Souffrance'].astype(str).str.strip().replace({'','nan','NaN','None'}, None)
    souff = cleaned.dropna()
    return len(souff), len(df)


def plot_souffrance_plotly(count, total):
    fig = go.Figure(data=[go.Pie(
        labels=['Avec Souffrance', 'Sans Souffrance'],
        values=[count, total - count],
        textinfo='percent+label',
        marker=dict(colors=[COLOR_ALERT, COLOR_PRIMARY]),
        hole=0.4
    )])
    fig.update_layout(title="Proportion des BL avec Souffrance",
                      height=400,
                      font=dict(color="black", size=12))
    return fig


def plot_livraison_kpi_plotly(df):
    nb_parties = df['Date_depart_dt'].notna().sum()
    nb_livrees = df['Date_liv_dt'].notna().sum()
    fig = go.Figure(data=[go.Pie(
        labels=['Livrées', 'Non livrées'],
        values=[nb_livrees, nb_parties - nb_livrees],
        textinfo='percent+label',
        marker=dict(colors=[COLOR_PRIMARY, COLOR_ALERT]),
        hole=0.3
    )])
    fig.update_layout(
        title="Taux de livraison",
        annotations=[dict(
            text=f"{nb_livrees}/{nb_parties} livrées",
            x=0.5, y=1.15, xref='paper', yref='paper', showarrow=False
        )],
        height=400,
        font=dict(color="black", size=12)
    )
    return fig


def plot_rdv_respect_plotly(df):
    counts = df['RDV_respect'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=['RDV Respecté', 'RDV Non Respecté'],
        values=[counts.get(True,0), counts.get(False,0)],
        textinfo='percent+label',
        marker=dict(colors=[COLOR_PRIMARY, COLOR_ALERT]),
        hole=0.4
    )])
    fig.update_layout(title="Taux de respect des RDV IAF/AFF",
                      height=400,
                      font=dict(color="black", size=12))
    return fig


# ==========================
# MAIN UI
# ==========================

st.title("📦 KPI Transport DIM")

# Sélecteur date
min_date = df['Date_BE_dt'].min().date()
max_date = df['Date_BE_dt'].max().date()
col_date, _ = st.columns([1, 3])
with col_date:
    date_range = st.date_input(
        "Période Date_BE",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

df_filtered = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (df_filtered['Date_BE_dt'] >= pd.to_datetime(start_date)) &
        (df_filtered['Date_BE_dt'] <= pd.to_datetime(end_date))
    ]

# --------------------------
# FILTRES SIDEBAR
# --------------------------

with st.sidebar:
    st.header("🔍 Filtres")
    if 'Type_Transport' in df_filtered:
        opts = sorted(df_filtered['Type_Transport'].dropna().unique())
        sel_multi = st.multiselect("🚛 Type Transport", options=opts, default=opts)
    if sel_multi:
        df_filtered = df_filtered[df_filtered['Type_Transport'].isin(sel_multi)]

    if 'CHRONO' in df_filtered:
        opts = df_filtered['CHRONO'].dropna().unique()
        sel2 = st.selectbox("⏱️ CHRONO", ["(Tous)"] + sorted(opts))
        if sel2 != "(Tous)":
            df_filtered = df_filtered[df_filtered['CHRONO'] == sel2]

# --------------------------
# TABLEAU
# --------------------------

st.subheader("📋 Données brutes")
df_display = df_filtered.drop(columns=['Date_BE_dt', 'Date_depart_dt', 'Date_liv_dt', 'Date_rdv_dt'], errors='ignore')
df_display = df_display.reset_index(drop=True)
st.dataframe(df_display, use_container_width=True)

# --------------------------
# GRAPHIQUES DELTA, SOUFFRANCE ET RDV EN COLONNE
# --------------------------

col_delta, col_souffrance_rdv = st.columns([1,1])

# Graphique DELTA
with col_delta:
    df_delta = df_filtered[df_filtered['Delta_jours_ouvres'].notna()]
    delta_series = df_delta['Delta_jours_ouvres'].astype(int)
    delta_series_capped = delta_series.apply(lambda x: x if x <= 7 else 7)
    delta_series_capped = delta_series_capped.replace({7: "7+"})

    if not delta_series_capped.empty:
        delta_counts = delta_series_capped.value_counts().sort_index(key=lambda x: [int(v.rstrip("+")) if isinstance(v, str) else v for v in x])
        st.subheader("📊 Répartition des délais de livraison")
        st.markdown(f"**{delta_counts.sum()} expéditions avec un délai mesuré**")
        st.plotly_chart(plot_delta_plotly(delta_counts), use_container_width=True, theme=None)
    else:
        st.info("Pas de données avec délai mesuré.")

# Graphiques SOUFFRANCE et RDV
with col_souffrance_rdv:
    col_souf, col_rdv = st.columns(2)

    # Souffrance
    with col_souf:
        souff_count, total_rows = count_souffrance(df_filtered)
        st.subheader("⚠️ Analyse Souffrance")
        st.markdown(f"**{souff_count} sur {total_rows} BL avec souffrance**")
        st.plotly_chart(plot_souffrance_plotly(souff_count, total_rows), use_container_width=True, theme=None)

    # RDV
    with col_rdv:
        if 'Type_Transport' in df_filtered.columns:
            df_rdv = df_filtered[df_filtered['Type_Transport'].isin(['IAF','AFF'])]
            df_rdv = df_rdv[df_rdv['Date_liv_dt'].notna() & df_rdv['Date_rdv_dt'].notna()]
            st.subheader("⏰ Taux de respect des RDV IAF/AFF")

            if not df_rdv.empty:
                df_rdv['RDV_respect'] = df_rdv['Date_liv_dt'] <= df_rdv['Date_rdv_dt']
                st.plotly_chart(plot_rdv_respect_plotly(df_rdv), use_container_width=True, theme=None)
                rrate = df_rdv['RDV_respect'].mean() * 100
                st.markdown(f"**Taux de respect : {rrate:.1f}% ({df_rdv['RDV_respect'].sum()} sur {len(df_rdv)})**")
            else:
                st.info("Aucune donnée avec Date Livraison et Date RDV pour produits IAF/AFF.")


# --------------------------
# KPI Livraison Mensuel
# --------------------------

# --------------------------
# KPI Livraison Mensuel avec moyenne glissante
# --------------------------

st.subheader("📅 KPI Livraison par mois")
df_liv = df_filtered[df_filtered['Date_depart_dt'].notna()]

if df_liv.empty:
    st.info("Aucune donnée avec Date départ pour calculer le KPI Livraison.")
else:
    df_liv['Mois'] = df_liv['Date_depart_dt'].dt.to_period('M').astype(str)
    df_liv['Livree'] = df_liv['Date_liv_dt'].notna()
    kpi_mensuel = df_liv.groupby('Mois').agg(nb_total=('Livree','count'), nb_livrees=('Livree','sum'))
    kpi_mensuel['taux'] = (kpi_mensuel['nb_livrees']/kpi_mensuel['nb_total'])*100

    # Tri par mois pour être sûr que c'est chronologique
    kpi_mensuel = kpi_mensuel.sort_index()

    # Moyenne glissante sur 3 mois
    kpi_mensuel['moyenne_glissante'] = kpi_mensuel['taux'].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()

    # Barres du taux par mois
    fig.add_trace(go.Bar(
        x=kpi_mensuel.index,
        y=kpi_mensuel['taux'],
        text=[f"{v:.1f}%" for v in kpi_mensuel['taux']],
        textposition='outside',
        marker_color=COLOR_PRIMARY,
        name="Taux mensuel"
    ))

    # Ligne de moyenne glissante
    fig.add_trace(go.Scatter(
        x=kpi_mensuel.index,
        y=kpi_mensuel['moyenne_glissante'],
        mode='lines+markers+text',
        name="Moyenne glissante 3 mois",
        line=dict(color=COLOR_ALERT, dash='dash'),
        text=[f"{v:.1f}%" for v in kpi_mensuel['moyenne_glissante']],
        textposition="top right"
    ))

    fig.update_layout(
        title="Taux de Livraison par Mois avec Moyenne Glissante",
        xaxis_title="Mois",
        yaxis_title="Taux de livraison (%)",
        height=450,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=60),
    )

    st.plotly_chart(fig, use_container_width=True, theme=None)



# --------------------------
# EXPORTS
# --------------------------

csv = df_display.to_csv(index=False).encode("utf-8")
st.download_button("📅 Export CSV", data=csv, file_name="export_dynamique.csv", mime="text/csv")

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
    df_display.to_excel(writer, index=False, sheet_name="Données")

st.download_button("📅 Export Excel",
                   data=excel_buf.getvalue(),
                   file_name="export_dynamique.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
