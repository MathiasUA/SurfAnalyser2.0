import streamlit as st
import pandas as pd
from datetime import datetime


# =============================================================================
# CONFIGURATIE & LOGICA
# =============================================================================

def bepaal_strand_drukte(mensen):
    if mensen < 5:
        return "Rustig 🟢"
    elif mensen < 20:
        return "Gemiddeld 🟡"
    else:
        return "Druk 🔴"


def bepaal_zee_drukte(mensen):
    if mensen < 3:
        return "Rustig 🟢"
    elif mensen <= 7:
        return "Gemiddeld 🟡"
    else:
        return "Druk 🔴"


def bepaal_weer(licht_pct):
    if licht_pct > 75:
        return "Onbewolkt ☀️"
    elif licht_pct > 50:
        return "Bewolkt (Wit) ⛅"
    elif licht_pct > 20:
        return "Bewolkt (Grijs) ☁️"
    else:
        return "Donker 🌙"


def bepaal_surf_omstandigheden(mensen_zee, foam_pct, wave_freq, licht_pct):
    if licht_pct <= 20:
        return "Slecht 🚫 (Te donker)"
    if mensen_zee > 7:
        return "Matig 😐 (Te druk in zee)"
    if foam_pct < 1.0 or wave_freq < 1.0:
        return "Matig 📉 (Weinig golven / Flat)"
    elif foam_pct > 3.0 and wave_freq >= 3.0:
        return "Perfect 🏄 (Top golven!)"
    else:
        return "Goed 👍 (Oké omstandigheden)"


# =============================================================================
# DASHBOARD OPBOUW
# =============================================================================

st.set_page_config(page_title="Surf Analytics Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("🏄‍♂️ Surf & Strand Analytics")
st.markdown("Live interpretatie van de cameradata via de IoT-sensor.")


@st.cache_data(ttl=5)
def load_data():
    try:
        df = pd.read_csv("surf_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.warning("Geen data gevonden. Laat je OpenCV sensor-script eerst draaien!")
else:
    laatste_rij = df.iloc[-1]

    status_strand = bepaal_strand_drukte(laatste_rij['people_beach'])
    status_zee = bepaal_zee_drukte(laatste_rij['people_water'])
    status_weer = bepaal_weer(laatste_rij['sky_brightness_pct'])
    status_surf = bepaal_surf_omstandigheden(
        laatste_rij['people_water'],
        laatste_rij['foam_coverage_pct'],
        laatste_rij['wave_freq_bpm'],
        laatste_rij['sky_brightness_pct']
    )

    # --- TOP ROW: KPI's ---
    st.subheader("Huidige Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Surf Omstandigheden", status_surf)
    with col2:
        st.metric("Drukte Strand", status_strand, f"{laatste_rij['people_beach']} personen")
    with col3:
        st.metric("Drukte Zee", status_zee, f"{laatste_rij['people_water']} personen")
    with col4:
        st.metric("Lucht / Weer", status_weer, f"{laatste_rij['sky_brightness_pct']}% felheid")

    st.markdown("---")

    # --- MIDDLE ROW: GRAFIEKEN ---
    st.subheader("Live Trends (Laatste metingen)")

    df_plot = df.tail(100).set_index('timestamp')
    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown("**🧍 Aantal personen op het strand**")
        st.line_chart(df_plot['people_beach'], color="#00ff00")

    with g_col2:
        st.markdown("**🏊 Aantal personen in zee**")
        st.line_chart(df_plot['people_water'], color="#0000ff")

    with g_col3:
        st.markdown("**🌊 Golffrequentie (Pieken/min)**")
        st.line_chart(df_plot['wave_freq_bpm'], color="#ff0000")

    st.markdown("---")

    # --- BOTTOM ROW: RUWE DATA & EXPORT ---
    st.subheader("Ruwe Sensor Data")

    # AANGEPAST: width='stretch' ipv het verouderde use_container_width
    st.dataframe(df.sort_values(by="timestamp", ascending=False), width="stretch", hide_index=True)

    csv_export = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Dataset als CSV",
        data=csv_export,
        file_name=f'surf_analytics_export_{datetime.today().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )