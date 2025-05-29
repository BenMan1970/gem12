import streamlit as st
import pandas as pd
import numpy as np
import time
import requests

st.set_page_config(page_title="Scanner Confluence Forex (TwelveData)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Analyse technique multi-indicateurs avec données TwelveData H1*")

# --- Gestion clé API ---
try:
    TWELVEDATA_API_KEY = st.secrets["TWELVEDATA_API_KEY"]
    st.sidebar.success("✅ Clé API TwelveData chargée")
except Exception as e:
    st.sidebar.error("❌ Clé API TwelveData manquante ou mal configurée.")
    st.error(f"**Configuration requise:** Ajoutez TWELVEDATA_API_KEY dans les secrets Streamlit. Erreur: {e}")
    st.stop()

FOREX_PAIRS_TWELVEDATA = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
    "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

# --- Fonctions indicateurs techniques ---
def safe_ewm_mean(series, **kwargs):
    if series.isna().all() or len(series.dropna()) < kwargs.get('span', kwargs.get('com', kwargs.get('halflife', 1))):
        return pd.Series(np.nan, index=series.index)
    return series.ewm(**kwargs).mean()

def ema(s, p): return safe_ewm_mean(s, span=p, adjust=False)
def rma(s, p): return safe_ewm_mean(s, alpha=1/p, adjust=False)

def hull_ma_pine(dc, p=20):
    try:
        required_length = p + int(np.sqrt(p))
        if len(dc.dropna()) < required_length:
            return pd.Series(np.nan, index=dc.index)
        hl = max(1, int(p/2))
        sl = max(1, int(np.sqrt(p)))
        if len(dc.dropna()) < hl or len(dc.dropna()) < p:
            return pd.Series(np.nan, index=dc.index)
        wma1 = dc.rolling(window=hl).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) == hl else np.nan,
            raw=True
        )
        wma2 = dc.rolling(window=p).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) == p else np.nan,
            raw=True
        )
        raw_hma = 2 * wma1 - wma2
        if len(raw_hma.dropna()) < sl:
            return pd.Series(np.nan, index=dc.index)
        hma = raw_hma.rolling(window=sl).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) == sl else np.nan,
            raw=True
        )
        return hma
    except Exception as e:
        st.error(f"Erreur dans hull_ma_pine: {e}")
        return pd.Series(np.nan, index=dc.index)

def rsi_pine(po4, p=10):
    try:
        if len(po4.dropna()) < (p + 1):
            return pd.Series(50.0, index=po4.index)
        d = po4.diff()
        g = d.where(d > 0, 0.0)
        l = -d.where(d < 0, 0.0)
        ag = rma(g, p)
        al = rma(l, p)
        al_safe = al.replace(0, 1e-9)
        rs = ag / al_safe
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        st.error(f"Erreur dans rsi_pine: {e}")
        return pd.Series(50.0, index=po4.index)

def adx_pine(h, l, c, p=14):
    try:
        if len(h.dropna()) < (p * 2) or len(l.dropna()) < (p * 2) or len(c.dropna()) < (p * 2):
            return pd.Series(0.0, index=h.index)
        # Debug: Check data types
        if not all(isinstance(x, (int, float)) for x in h.dropna()) or \
           not all(isinstance(x, (int, float)) for x in l.dropna()) or \
           not all(isinstance(x, (int, float)) for x in c.dropna()):
            st.error(f"Erreur dans adx_pine: Données non numériques détectées - High: {h.dtype}, Low: {l.dtype}, Close: {c.dtype}")
            return pd.Series(0.0, index=h.index)
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rma(tr, p)
        um = h.diff()
        dm = l.shift(1) - l
        pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=h.index)
        mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=h.index)
        satr = atr.replace(0, 1e-9)
        pdi = 100 * (rma(pdm, p) / satr)
        mdi = 100 * (rma(mdm, p) / satr)
        dxden = (pdi + mdi).replace(0, 1e-9)
        dx = 100 * (abs(pdi - mdi) / dxden)
        return rma(dx, p).fillna(0)
    except Exception as e:
        st.error(f"Erreur dans adx_pine: {e}")
        return pd.Series(0.0, index=h.index)

def heiken_ashi_pine(dfo):
    try:
        if len(dfo.dropna()) < 1:
            return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
        # Compute HA_Close as a Series directly
        ha_close = (dfo['Open'] + dfo['High'] + dfo['Low'] + dfo['Close']) / 4
        ha_open = pd.Series(np.nan, index=dfo.index)
        if len(dfo) > 0:
            ha_open.iloc[0] = (dfo['Open'].iloc[0] + dfo['Close'].iloc[0]) / 2
        for i in range(1, len(dfo)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        return ha_open, ha_close
    except Exception as e:
        st.error(f"Erreur dans heiken_ashi_pine: {e}")
        return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)

def smoothed_heiken_ashi_pine(dfo, l1=10, l2=10):
    try:
        if len(dfo.dropna()) < (max(l1, l2) + 1):
            return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
        eo = ema(dfo['Open'], l1)
        eh = ema(dfo['High'], l1)
        el = ema(dfo['Low'], l1)
        ec = ema(dfo['Close'], l1)
        hai = pd.DataFrame({'Open': eo, 'High': eh, 'Low': el, 'Close': ec}, index=dfo.index).dropna()
        if len(hai) == 0:
            return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
        hao_i, hac_i = heiken_ashi_pine(hai)
        sho = ema(hao_i, l2)
        shc = ema(hac_i, l2)
        return sho, shc
    except Exception as e:
        st.error(f"Erreur dans smoothed_heiken_ashi_pine: {e}")
        return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    try:
        min_len_req = max(tenkan_p, kijun_p, senkou_b_p)
        if len(df_high.dropna()) < min_len_req or len(df_low.dropna()) < min_len_req or len(df_close.dropna()) < min_len_req:
            return 0
        tenkan_sen = (df_high.rolling(window=tenkan_p).max() + df_low.rolling(window=tenkan_p).min()) / 2
        kijun_sen = (df_high.rolling(window=kijun_p).max() + df_low.rolling(window=kijun_p).min()) / 2
        if tenkan_sen.iloc[-1] > kijun_sen.iloc[-1] and df_close.iloc[-1] > kijun_sen.iloc[-1]:
            return 1
        elif tenkan_sen.iloc[-1] < kijun_sen.iloc[-1] and df_close.iloc[-1] < kijun_sen.iloc[-1]:
            return -1
        return 0
    except Exception as e:
        st.error(f"Erreur dans ichimoku_pine_signal: {e}")
        return 0

# --- Fonction de récupération des données Twelve Data ---
@st.cache_data(ttl=300)
def get_data_twelvedata(pair_symbol, interval_td="1h", outputsize_td=300):
    try:
        url = (
            f"https://api.twelvedata.com/time_series?"
            f"symbol={pair_symbol}&interval={interval_td}&outputsize={outputsize_td}&apikey={TWELVEDATA_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            st.warning(f"⚠️ Limite de requêtes atteinte pour {pair_symbol}. Attendez et réessayez.")
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "error":
            st.warning(f"Erreur TwelveData pour {pair_symbol}: {data.get('message', 'Aucune donnée reçue')}")
            return None
        if "values" not in data or not data["values"]:
            st.warning(f"Aucune donnée pour {pair_symbol}. Vérifiez le symbole ou la disponibilité.")
            return None
        df = pd.DataFrame(data["values"])
        required_cols = ["open", "high", "low", "close", "datetime"]
        if not all(col in df.columns for col in required_cols):
            st.warning(f"Données incomplètes pour {pair_symbol}. Colonnes manquantes.")
            return None
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        df = df[["datetime", "Open", "High", "Low", "Close"]].sort_values("datetime").reset_index(drop=True)
        if df.empty or df["Close"].isna().all():
            st.warning(f"Données vides ou invalides pour {pair_symbol}.")
            return None
        # Debug: Check data types after conversion
        st.write(f"Types de données pour {pair_symbol}: {df.dtypes}")
        return df
    except requests.exceptions.RequestException as e:
        st.warning(f"Erreur réseau pour {pair_symbol}: {e}")
        return None
    except Exception as e:
        st.warning(f"Erreur inattendue pour {pair_symbol}: {e}")
        return None

# --- Calcul multi-indicateurs Pine ---
def calculate_all_signals_pine(data):
    signal_details_pine = {}
    bull_confluences = 0
    bear_confluences = 0
    close = data['Close']
    high = data['High']
    low = data['Low']
    ohlc4 = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4

    # HMA
    try:
        hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
           nts = 0
            hma_val = hma_series.iloc[-1]
            hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev:
                bull_confluences += 1
                signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev:
                bear_confluences += 1
                signal_details_pine['HMA'] = "▼"
            else:
                signal_details_pine['HMA'] = "─"
        else:
            signal_details_pine['HMA'] = "N/A"
    except Exception as e:
        signal_details_pine['HMA'] = "ErrHMA"
        st.error(f"Erreur HMA: {e}")

    # RSI
    try:
        rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >= 1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]
            signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50:
                bull_confluences += 1
                signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50:
                bear_confluences += 1
                signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else:
                signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else:
            signal_details_pine['RSI'] = "N/A"
    except Exception as e:
        signal_details_pine['RSI'] = "ErrRSI"
        signal_details_pine['RSI_val'] = "N/A"
        st.error(f"Erreur RSI: {e}")

    # ADX
    try:
        adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]
            signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20:
                bull_confluences += 1
                bear_confluences += 1
                signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else:
                signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else:
            signal_details_pine['ADX'] = "N/A"
    except Exception as e:
        signal_details_pine['ADX'] = "ErrADX"
        signal_details_pine['ADX_val'] = "N/A"
        st.error(f"Erreur ADX: {e}")

    # Heiken Ashi
    try:
        ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >= 1 and len(ha_close) >= 1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]:
                bullconde = 0
                bull_confluences += 1
                signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['HA'] = "▼"
            else:
                signal_details_pine['HA'] = "─"
        else:
            signal_details_pine['HA'] = "N/A"
    except Exception as e:
        signal_details_pine['HA'] = "ErrHA"
        st.error(f"Erreur HA: {e}")

    # Smoothed Heiken Ashi
    try:
        sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >= 1 and len(sha_close) >= 1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]:
                bull_confluences += 1
                signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                bear_confluences += 1
                signal_details_pine['SHA'] = "▼"
            else:
                signal_details_pine['SHA'] = "─"
        else:
            signal_details_pine['SHA'] = "N/A"
    except Exception as e:
        signal_details袋['SHA'] = "ErrSHA"
        st.error(f"Erreur SHA: {e}")

    # Ichimoku
    try:
        ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1:
            bull_confluences += 1
            signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1:
            bear_confluences += 1
            signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9, 26, 52) or (len(data) > 0 and pd.isna(data['Close'].iloc[-1]))):
            signal_details_pine['Ichi'] = "N/D"
        else:
            signal_details_pine['Ichi'] = "─"
    except Exception as e:
        signal_details_pine['Ichi'] = "ErrIchi"
        st.error(f"Erreur Ichi: {e}")

    confluence_value = max(bull_confluences, bear_confluences)
    direction = "NEUTRE"
    if bull_confluences > bear_confluences:
        direction = "HAUSSIER"
    elif bear_confluences > bull_confluences:
        direction = "BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0:
        direction = "CONFLIT"

    return {
        'confluence_P': confluence_value,
        'direction_P': direction,
        'bull_P': bull_confluences,
        'bear_P': bear_confluences,
        'rsi_P': signal_details_pine.get('RSI_val', "N/A"),
        'adx_P': signal_details_pine.get('ADX_val', "N/A"),
        'signals_P': signal_details_pine
    }

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"
    elif confluence_value == 5: return "⭐⭐⭐⭐⭐"
    elif confluence_value == 4: return "⭐⭐⭐⭐"
    elif confluence_value == 3: return "⭐⭐⭐"
    elif confluence_value == 2: return "⭐⭐"
    elif confluence_value == 1: return "⭐"
    else: return "WAIT"

# --- Interface Utilisateur ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    min_conf = st.selectbox("Confluence minimum", (0, 1, 2, 3, 4, 5, 6), index=3, format_func=lambda x: f"{x} étoiles")
    show_all = st.checkbox("Afficher toutes les paires", True)
    scan_btn = st.button("🔍 Lancer le Scan", type="primary", use_container_width=True, disabled=not TWELVEDATA_API_KEY, help="Scanner les paires Forex")

with col2:
    if scan_btn and TWELVEDATA_API_KEY:
        st.info("🔄 Scan en cours...")
        pr_res = []
        pb = st.progress(0)
        stx = st.empty()
        total_pairs = len(FOREX_PAIRS_TWELVEDATA)
        for i, pair_symbol in enumerate(FOREX_PAIRS_TWELVEDATA):
            pnd = pair_symbol
            cp = (i + 1) / total_pairs
            pb.progress(cp)
            stx.text(f"Analyse: {pnd} ({i + 1}/{total_pairs})")
            d_h1_td = get_data_twelvedata(pair_symbol, interval_td="1h", outputsize_td=300)
            if d_h1_td is None or d_h1_td.empty:
                pr_res.append({
                    'Paire': pnd, 'Direction': 'ERR DATA', 'Conf. (0-6)': 0,
                    'Étoiles': venuta'A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull': 0, 'Bear': 0,
                    'details': {'Info': 'Données TwelveData non disponibles'}
                })
                continue
            sigs = calculate_all_signals_pine(d_h1_td)
            if sigs:
                strs = get_stars_pine(sigs['confluence_P'])
                rd = {
                    'Paire': pnd,
                    'Direction': sigs['direction_P'],
                    'Conf. (0-6)': sigs['confluence_P'],
                    'Étoiles': strs,
                    'RSI': sigs['rsi_P'],
                    'ADX': sigs['adx_P'],
                    'Bull': sigs['bull_P'],
                    'Bear': sigs['bear_P'],
                    'details': sigs['signals_P']
                }
                pr_res.append(rd)
            else:
                pr_res.append({
                    'Paire': pnd, 'Direction': 'ERR CALC', 'Conf. (0-6)': 0,
                    'Étoiles': 'N/A', 'RSI': 'N/A', 'ADX': 'N/A', 'Bull': 0, 'Bear': 0,
                    'details': {'Info': 'Calcul échoué'}
                })
            if i < total_pairs - 1:
                time.sleep(8)  # Respecter la limite de 8 appels par minute
        pb.empty()
        stx.empty()
        if pr_res:
            dfa = pd.DataFrame(pr_res)
            dfd = dfa[dfa['Conf. (0-6)'] >= min_conf].copy() if not show_all else dfa.copy()
            if not show_all:
                st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence.")
            else:
                st.info(f"🔍 Affichage {len(dfd)} paires.")
            if not dfd.empty:
                dfds = dfd.sort_values('Conf. (0-6)', ascending=False)
                vcs = [c for c in ['Paire', 'Direction', 'Conf. (0-6)', 'Étoiles', 'RSI', 'ADX', 'Bull', 'Bear'] if c in dfds.columns]
                st.dataframe(dfds[vcs], use_container_width=True, hide_index=True)
                with st.expander("📊 Détails des signaux"):
                    for _, r in dfds.iterrows():
                        sm = r.get('details', {})
                        if not isinstance(sm, dict):
                            sm = {'Info': 'Détails non disponibles'}
                        st.write(f"**{r.get('Paire', 'N/A')}** - {r.get('Étoiles', 'N/A')} ({r.get('Conf. (0-6)', 'N/A')}) - Dir: {r.get('Direction', 'N/A')}")
                        dc = st.columns(6)
                        so = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                        for idx, sk in enumerate(so):
                            dc[idx].metric(label=sk, value=sm.get(sk, "N/P"))
                        st.divider()
            else:
                st.warning("❌ Aucune paire ne répond aux critères. Vérifiez les erreurs.")
        else:
            st.error("❌ Aucune paire traitée. Vérifiez les logs serveur.")
    else:
        st.info("Cliquez sur 'Lancer le Scan' pour analyser les paires Forex.")

with st.expander("ℹ️ Informations"):
    st.markdown("""**Signaux:** HMA(20), RSI(10), ADX(14)>=20, HA, SHA(10,10), Ichi(9,26,52). **Source:** TwelveData API.""")
    st.caption("Scanner H1 (TwelveData). Respectez les limites de l'API gratuite (8 req/min).")
       
