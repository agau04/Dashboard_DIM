import streamlit as st
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Statistiques DIM", layout="wide")

# --- Couleurs & Styles ---
COLOR_PRIMARY = "#2BC559"  # Bleu
COLOR_ALERT = "#C72B2E"    # Rouge
BACKGROUND_COLOR = "#FFFFFF"  # Fond clair pour les graphes

# 📌 Bouton de refresh dans la sidebar, dans une interaction
with st.sidebar:
    if st.button("🔁 Recharger les données"):
        st.cache_data.clear()
        st.experimental_rerun()  # st.rerun() deprecated, experimental_rerun est stable

@st.cache_data(ttl=600)  # Cache 10 min
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
        df = df.dropna(subset=['Date_BE_dt'])
    return df

@st.cache_data(ttl=300)
def clean_delta_column(df: pd.DataFrame):
    if 'Delta' not in df.columns:
        return pd.Series(dtype='float64')
    delta_clean = (
        df['Delta']
        .astype(str)
        .str.strip()
        .replace({'--': None, 'NC': None, '': None, 'nan': None, 'NaN': None, 'None': None})
    )
    delta_clean = pd.to_numeric(delta_clean, errors='coerce')
    return delta_clean.dropna()

@st.cache_data(ttl=300)
def count_souffrance(df: pd.DataFrame):
    if 'Souffrance' not in df.columns:
        return 0, 0
    souffrance_non_null = (
        df['Souffrance']
        .astype(str)
        .str.strip()
        .replace({'', 'nan', 'NaN', 'None'}, None)
        .dropna()
    )
    return len(souffrance_non_null), len(df)

def plot_delta(delta_counts):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BACKGROUND_COLOR)
    bars = ax.bar(delta_counts.index.astype(str), delta_counts.values, color=COLOR_PRIMARY, edgecolor='none')

    ax.set_xlabel('Délai de livraison (jours)', fontsize=10, color='#333')
    ax.set_ylabel("Nombre d'expéditions", fontsize=10, color='#333')
    ax.set_title('Répartition des délais de livraison', fontsize=12, color='#222')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('#AAA')
    ax.spines['bottom'].set_color('#AAA')

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(delta_counts.values)*1.15)

    ax.tick_params(axis='x', colors='#555', labelsize=9)
    ax.tick_params(axis='y', colors='#555', labelsize=9)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8,
                    color=COLOR_PRIMARY,
                    fontweight='bold')

    fig.tight_layout()
    return fig

def plot_souffrance(count_souffrance, total):
    labels = ['Avec Souffrance', 'Sans Souffrance']
    sizes = [count_souffrance, total - count_souffrance]
    colors = [COLOR_ALERT, COLOR_PRIMARY]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BACKGROUND_COLOR)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10, 'color': '#333'}
    )
    for txt in texts + autotexts:
        txt.set_fontsize(10)
        txt.set_color('#444')

    ax.axis('equal')
    ax.set_title("Proportion des BL avec Souffrance", fontsize=12, color='#222')

    fig.tight_layout()
    return fig

# --- MAIN ---

st.title("📦 KPI Transport DIM")

df = load_csv_from_url()

if df.empty:
    st.warning("Aucune donnée chargée pour l'instant.")
    st.stop()

df = preprocess_df(df)

df_filtered = df.copy()

# 🎛️ Filtres dans sidebar
with st.sidebar:
    st.header("🔍 Filtres")

    if 'Date_BE_dt' in df_filtered.columns and not df_filtered.empty:
        min_date = df_filtered['Date_BE_dt'].min().date()
        max_date = df_filtered['Date_BE_dt'].max().date()
        date_range = st.date_input(
            "📅 Période Date_BE",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[
                (df_filtered['Date_BE_dt'] >= pd.to_datetime(start_date)) &
                (df_filtered['Date_BE_dt'] <= pd.to_datetime(end_date))
            ]
        else:
            st.warning("Veuillez sélectionner une **plage de deux dates**.")

    if 'Type_Transport' in df_filtered.columns:
        uniques = df_filtered['Type_Transport'].dropna().unique()
        if len(uniques) > 0:
            selected = st.selectbox(
                "🚛 Type Transport",
                options=["(Tous)"] + sorted(uniques)
            )
            if selected != "(Tous)":
                df_filtered = df_filtered[df_filtered['Type_Transport'] == selected]

# 🧾 Tableau principal
st.subheader("📋 Données brut")
df_display = df_filtered.drop(columns=['Date_BE_dt'], errors='ignore').reset_index(drop=True)
st.dataframe(df_display, use_container_width=True)

col1, col2 = st.columns(2)

# ➤ Graphique Delta
delta_non_null = clean_delta_column(df_filtered)

if len(delta_non_null) > 0:
    delta_counts = delta_non_null.value_counts().sort_index()
    delta_counts = delta_counts[delta_counts.index <= 30]

    with col1:
        st.subheader("📊 Répartition des délais de livraison (Delta)")
        st.markdown(f"**{len(delta_non_null)} BL** livrés avec un délai mesuré")

        fig = plot_delta(delta_counts)
        st.pyplot(fig)
else:
    with col1:
        st.info("La colonne 'Delta' ne contient pas de valeurs valides ou est absente.")

# ➤ Graphique Souffrance
count_souffrance_val, total_rows = count_souffrance(df_filtered)

if total_rows > 0 and 'Souffrance' in df_filtered.columns:
    with col2:
        st.subheader("⚠️ Analyse Souffrance")
        st.markdown(f"**{count_souffrance_val} BL** sur **{total_rows}** ont une mention Souffrance")

        fig2 = plot_souffrance(count_souffrance_val, total_rows)
        st.pyplot(fig2)
else:
    with col2:
        st.info("Colonne 'Souffrance' absente ou aucune donnée analysable.")

# 📤 Export CSV / Excel
csv = df_display.to_csv(index=False).encode('utf-8')
st.download_button("📥 Export CSV", data=csv, file_name='export_dynamique.csv', mime='text/csv')

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    df_display.to_excel(writer, sheet_name='Données', index=False)
st.download_button(
    "📥 Export Excel",
    data=excel_buffer.getvalue(),
    file_name='export_dynamique.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
