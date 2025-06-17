import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt
import holidays

st.set_page_config(page_title="Statistiques DIM", layout="wide")

# --- Couleurs & Styles ---
COLOR_PRIMARY = "#619CDA"
COLOR_ALERT = '#E15759'
BACKGROUND_COLOR = '#F9F9F9'

# Initialiser les jours fériés France
fr_holidays = holidays.France(years=range(2020, 2031))

# Bouton refresh
with st.sidebar:
    if st.button("🔁 Recharger les données"):
        st.cache_data.clear()
        st.experimental_rerun()

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
def preprocess_df(df: pd.DataFrame):
    if 'Date_BE' in df.columns:
        df['Date_BE_dt'] = pd.to_datetime(df['Date_BE'], errors='coerce')
    if 'Date_depart' in df.columns:
        df['Date_depart_dt'] = pd.to_datetime(df['Date_depart'], errors='coerce')
    if 'Date_liv' in df.columns:
        df['Date_liv_dt'] = pd.to_datetime(df['Date_liv'], errors='coerce')
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
def calculate_delta_jours_ouvres(df: pd.DataFrame):
    df['Delta_jours_ouvres'] = df.apply(
        lambda row: jours_ouvres(row['Date_depart_dt'], row['Date_liv_dt']) if pd.notna(row['Date_depart_dt']) and pd.notna(row['Date_liv_dt']) else np.nan,
        axis=1
    )
    return df

@st.cache_data(ttl=300)
def clean_delta_column(df: pd.DataFrame):
    if 'Delta' not in df.columns:
        return pd.Series(dtype='float64')
    delta_clean = df['Delta'].astype(str).str.strip().replace({'--': None, 'NC': None, '': None, 'nan': None, 'NaN': None, 'None': None})
    delta_clean = pd.to_numeric(delta_clean, errors='coerce')
    return delta_clean.dropna()

@st.cache_data(ttl=300)
def count_souffrance(df: pd.DataFrame):
    if 'Souffrance' not in df.columns:
        return 0, 0
    souffrance_non_null = df['Souffrance'].astype(str).str.strip().replace({'', 'nan', 'NaN', 'None'}, None).dropna()
    return len(souffrance_non_null), len(df)

@st.cache_data(ttl=300)
def extract_departements(df: pd.DataFrame):
    if 'CP' in df.columns:
        df['Departement'] = df['CP'].astype(str).str[:2]
        df = df[df['Departement'].str.match(r'^\d{2}$')]
    else:
        df['Departement'] = None
    return df

def plot_delta(delta_counts):
    total = delta_counts.sum()
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BACKGROUND_COLOR)
    bars = ax.bar(delta_counts.index.astype(str), delta_counts.values, color=COLOR_PRIMARY)

    ax.set_xlabel('Délai de livraison (jours)', fontsize=10, color='#333')
    ax.set_ylabel("Nombre d'expéditions", fontsize=10, color='#333')
    ax.set_title('Répartition des délais de livraison', fontsize=12, fontweight='bold', color='#222')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(delta_counts.values)*1.25)

    for bar in bars:
        height = bar.get_height()
        pct = height / total * 100
        ax.annotate(f'{int(height)}\n({pct:.0f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    color=COLOR_PRIMARY,
                    fontweight='bold')
    fig.tight_layout()
    return fig

def plot_souffrance(count, total):
    labels = ['Avec Souffrance', 'Sans Souffrance']
    sizes = [count, total - count]
    colors = [COLOR_ALERT, COLOR_PRIMARY]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BACKGROUND_COLOR)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
           textprops={'fontsize': 10, 'color': '#333', 'fontweight': 'bold'})
    ax.axis('equal')
    ax.set_title("Proportion des BL avec Souffrance", fontsize=12, fontweight='bold', color='#222')
    fig.tight_layout()
    return fig

def plot_souffrance_motifs(df):
    if 'Souffrance' not in df.columns:
        st.info("La colonne 'Souffrance' est absente.")
        return None

    # Nettoyer les valeurs vides / nulles
    motifs = df['Souffrance'].astype(str).str.strip()
    motifs = motifs.replace({'': None, 'nan': None, 'NaN': None, 'None': None})
    motifs = motifs.dropna()

    if motifs.empty:
        st.info("Pas de motifs de souffrance valides dans les données.")
        return None

    counts = motifs.value_counts()

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BACKGROUND_COLOR)
    bars = ax.bar(counts.index, counts.values, color=COLOR_ALERT)
    ax.set_xlabel("Motifs de souffrance", fontsize=10, color='#333')
    ax.set_ylabel("Nombre d'occurrences", fontsize=10, color='#333')
    ax.set_title("Répartition des motifs de souffrance", fontsize=12, fontweight='bold', color='#222')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(counts.values)*1.2)
    plt.xticks(rotation=45, ha='right')

    # Affichage des valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    color=COLOR_ALERT,
                    fontweight='bold')
    fig.tight_layout()
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
        date_range = st.date_input("📅 Période Date_BE", value=[min_date, max_date], min_value=min_date, max_value=max_date)
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

df_filtered = extract_departements(df_filtered)

st.subheader("📋 Données brutes")
df_display = df_filtered.drop(columns=['Date_BE_dt', 'Date_depart_dt', 'Date_liv_dt'], errors='ignore').reset_index(drop=True)
st.dataframe(df_display, use_container_width=True)

col1, col2 = st.columns(2)

if 'Date_liv' in df_filtered:
    df_delta = df_filtered[df_filtered['Date_liv'].notna()]
else:
    df_delta = df_filtered

# Utiliser la colonne Delta_jours_ouvres au lieu de Delta
delta_series = df_delta['Delta_jours_ouvres'].dropna().astype(int)

if not delta_series.empty:
    delta_counts = delta_series.value_counts().sort_index()
    delta_counts = delta_counts[delta_counts.index <= 30]
    with col1:
        st.subheader("📊 Répartition des délais de livraison (jours ouvrés)")
        st.markdown(f"{len(delta_series)} BL avec un délai mesuré (jours ouvrés).")
        st.pyplot(plot_delta(delta_counts))
else:
    with col1:
        st.info("Pas de données avec délai mesuré.")


df_souffrance = df_filtered[df_filtered.get('Date_depart', pd.Series([True]*len(df_filtered))).notna()]
souff_count, total_rows = count_souffrance(df_souffrance)
if total_rows > 0:
    with col2:
        st.subheader("⚠️ Analyse Souffrance")
        st.markdown(f"{souff_count} sur {total_rows} BL avec souffrance.")
        st.pyplot(plot_souffrance(souff_count, total_rows))
else:
    with col2:
        st.info("Pas de données analysables pour la souffrance.")

if total_rows > 0:
    with col2:
        st.subheader("⚠️ Analyse Souffrance")
        st.markdown(f"{souff_count} sur {total_rows} BL avec souffrance.")
        
        # Afficher uniquement les motifs sous forme de barres
        fig_motifs = plot_souffrance_motifs(df_souffrance)
        if fig_motifs:
            st.pyplot(fig_motifs)
else:
    with col2:
        st.info("Pas de données analysables pour la souffrance.")



csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button("📥 Export CSV", data=csv, file_name='export_dynamique.csv', mime='text/csv')

excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
    df_display.to_excel(writer, sheet_name='Données', index=False)

st.download_button("📥 Export Excel", data=excel_buf.getvalue(), file_name='export_dynamique.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
