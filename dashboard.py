import streamlit as st
import pandas as pd
import io
import requests
import matplotlib.pyplot as plt

st.set_page_config(page_title="Statistiques DIM", layout="wide")

# 📌 Mettre le bouton de refresh ici dans la sidebar AVANT le chargement
with st.sidebar:
    if st.button("🔁 Recharger les données"):
        st.cache_data.clear()
        st.experimental_rerun()

@st.cache_data
def load_csv_from_url():
    url = "https://sobotram.teliway.com:443/appli/vsobotram/main/extraction.php?sAction=export&idDo=173&sCle=KPI_DIM&sTypeResultat=csv"
    response = requests.get(url, verify=False)
    response.raise_for_status()
    try:
        content = response.content.decode('utf-8')
    except UnicodeDecodeError:
        content = response.content.decode('iso-8859-1')
    df = pd.read_csv(io.StringIO(content), sep=';', engine='python')
    df.columns = [col.strip() for col in df.columns]
    return df


st.title("📦 Statistiques DIM (Sobotram)")

df = load_csv_from_url()

# Transformation date
if 'Date_BE' in df.columns:
    df['Date_BE_dt'] = pd.to_datetime(df['Date_BE'], errors='coerce')
    df = df.dropna(subset=['Date_BE_dt'])

df_filtered = df.copy()

# 🎛️ Filtres
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
                "Type_Transport",
                options=["(Tous)"] + sorted(uniques)
            )
            if selected != "(Tous)":
                df_filtered = df_filtered[df_filtered['Type_Transport'] == selected]

# 🧾 Tableau principal
st.subheader("📋 Données filtrées")
df_display = df_filtered.drop(columns=['Date_BE_dt'], errors='ignore').reset_index(drop=True)
st.dataframe(df_display, use_container_width=True)

# 🎨 Couleurs globales
COLOR_PRIMARY = '#4E79A7'  # Bleu
COLOR_ALERT = '#E15759'    # Rouge

# 📊 Graphiques
col1, col2 = st.columns(2)

# ➤ Graphique Delta
if 'Delta' in df_filtered.columns:
    df_filtered['Delta_clean'] = (
        df_filtered['Delta']
        .astype(str)
        .str.strip()
        .replace({'--': None, 'NC': None, '': None, 'nan': None, 'NaN': None, 'None': None})
    )
    df_filtered['Delta_clean'] = pd.to_numeric(df_filtered['Delta_clean'], errors='coerce')
    delta_non_null = df_filtered['Delta_clean'].dropna()

    if len(delta_non_null) > 0:
        delta_counts = delta_non_null.value_counts().sort_index()
        delta_counts = delta_counts[delta_counts.index <= 30]

        with col1:
            st.subheader("📊 Répartition des délais de livraison (Delta)")

            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(delta_counts.index.astype(str), delta_counts.values, color=COLOR_PRIMARY)

            ax.set_xlabel('Délai de livraison (jours)', fontsize=9)
            ax.set_ylabel("Nombre d'expéditions", fontsize=9)
            ax.set_title('Répartition des délais de livraison', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, max(delta_counts.values)*1.1)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

            st.pyplot(fig)
    else:
        with col1:
            st.info("La colonne 'Delta' ne contient pas de valeurs valides.")
else:
    with col1:
        st.info("Colonne 'Delta' absente.")

# ➤ Graphique Souffrance
if 'Souffrance' in df_filtered.columns:
    souffrance_non_null = df_filtered['Souffrance'].astype(str).str.strip().replace({'', 'nan', 'NaN', 'None'}, None).dropna()
    count_souffrance = len(souffrance_non_null)
    total = len(df_filtered)

    if total > 0:
        with col2:
            st.subheader("⚠️ Analyse Souffrance")
            st.markdown(f"**{count_souffrance} BL** sur **{total}** ont une mention Souffrance")

            labels = ['Avec Souffrance', 'Sans Souffrance']
            sizes = [count_souffrance, total - count_souffrance]
            colors = [COLOR_ALERT, COLOR_PRIMARY]

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 8}
            )
            for txt in texts + autotexts:
                txt.set_fontsize(8)
            ax2.axis('equal')
            ax2.set_title("Proportion des BL avec Souffrance", fontsize=9)

            st.pyplot(fig2)
    else:
        with col2:
            st.info("Aucune donnée analysable dans la colonne 'Souffrance'.")
else:
    with col2:
        st.info("Colonne 'Souffrance' absente.")

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
