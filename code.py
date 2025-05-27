import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback
import requests

st.set_page_config(page_title="Scanner Confluence Forex (TwelveData)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Analyse technique multi-indicateurs avec données TwelveData H1*")

# Gestion robuste de la clé API
try:
    TWELVEDATA_API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
    if TWELVEDATA_API_KEY:
        st.sidebar.success("✅ Clé API TwelveData chargée")
    else:
        st.sidebar.error("❌ Clé API TwelveData manquante dans les secrets.")
        st.error("**Configuration requise:** Ajoutez TWELVEDATA_API_KEY dans les secrets Streamlit.")
        st.stop()
except Exception as e:
    st.sidebar.error("❌ Erreur de configuration des secrets.")
    st.error(f"Vérifiez la configuration des secrets Streamlit: {e}")
    st.stop()

FOREX_PAIRS_TWELVEDATA = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

# ... (toutes vos fonctions indicateurs techniques ici, inchangées, mais bien indentées) ...

# Bloc paramètres utilisateur
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox("Confluence minimum", (0,1,2,3,4,5,6), index=3, format_func=lambda x: f"{x} étoiles")
    show_all = st.checkbox("Afficher toutes les paires", True)
    scan_dis_td = TWELVEDATA_API_KEY is None
    scan_tip_td = "Clé TwelveData non configurée." if scan_dis_td else "Scanner les paires Forex"
    scan_btn = st.button("🔍 Lancer le Scan", type="primary", use_container_width=True, disabled=scan_dis_td, help=scan_tip_td)

with col2:
    if scan_btn and TWELVEDATA_API_KEY:
        st.info("🔄 Scan en cours...")
        pr_res = []
        pb = st.progress(0)
        stx = st.empty()
        total_pairs = len(FOREX_PAIRS_TWELVEDATA)
        for i, pair_symbol in enumerate(FOREX_PAIRS_TWELVEDATA):
            pnd = pair_symbol
            cp = (i+1)/total_pairs
            pb.progress(cp)
            stx.text(f"Analyse: {pnd} ({i+1}/{total_pairs})")
            d_h1_td = get_data_twelvedata(pair_symbol, interval_td="1h", outputsize_td=300)
            if d_h1_td is not None and not d_h1_td.empty:
                sigs = calculate_all_signals_pine(d_h1_td)
                if sigs:
                    strs = get_stars_pine(sigs['confluence_P'])
                    rd = {'Paire': pnd, 'Direction': sigs['direction_P'], 'Conf. (0-6)': sigs['confluence_P'],
                          'Étoiles': strs, 'RSI': sigs['rsi_P'], 'ADX': sigs['adx_P'],
                          'Bull': sigs['bull_P'], 'Bear': sigs['bear_P'], 'details': sigs['signals_P']}
                    pr_res.append(rd)
                else:
                    pr_res.append({'Paire': pnd, 'Direction': 'ERR CALC', 'Conf. (0-6)': 0,
                                   'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull': 0, 'Bear': 0,
                                   'details': {'Info': 'Calcul échoué (TD)'}})
            else:
                pr_res.append({'Paire': pnd, 'Direction': 'ERR DATA', 'Conf. (0-6)': 0,
                               'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull': 0, 'Bear': 0,
                               'details': {'Info': 'Données TD non dispo (logs)'}})
            if i < total_pairs - 1:
                print(f"Pause 8s pour limite TD...")
                time.sleep(8)
        pb.empty()
        stx.empty()
        if pr_res:
            dfa = pd.DataFrame(pr_res)
            dfd = dfa[dfa['Conf. (0-6)'] >= min_conf].copy() if not show_all else dfa.copy()
            if not show_all:
                st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (TD).")
            else:
                st.info(f"🔍 Affichage {len(dfd)} paires (TD).")
            if not dfd.empty:
                dfds = dfd.sort_values('Conf. (0-6)', ascending=False)
                vcs = [c for c in ['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear'] if c in dfds.columns]
                st.dataframe(dfds[vcs], use_container_width=True, hide_index=True)
                with st.expander("📊 Détails des signaux (TD)"):
                    for _, r in dfds.iterrows():
                        sm = r.get('details', {})
                        if not isinstance(sm, dict):
                            sm = {'Info': 'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc = st.columns(6)
                        so = ['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx, sk in enumerate(so):
                            dc[idx].metric(label=sk, value=sm.get(sk, "N/P"))
                        st.divider()
            else:
                st.warning(f"❌ Aucune paire avec critères (TD). Vérifiez erreurs.")
        else:
            st.error("❌ Aucune paire traitée (TD). Vérifiez logs serveur.")

with st.expander("ℹ️ Informations"):
    st.markdown("""**Signaux:** HMA(20),RSI(10),ADX(14)>=20,HA,SHA(10,10),Ichi(9,26,52). **Source:** TwelveData API.""")
    st.caption("Scanner H1 (TwelveData). Respectez limites API.")
