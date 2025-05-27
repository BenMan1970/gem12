import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import traceback 
import requests 
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Scanner Confluence Forex (TwelveData)", 
    page_icon="⭐", 
    layout="wide"
)

st.title("🔍 Scanner Confluence Forex Premium")
st.markdown("*Analyse technique multi-indicateurs avec données TwelveData H1*")

# Gestion de la clé API
TWELVEDATA_API_KEY = None

try:
    TWELVEDATA_API_KEY = st.secrets.get("TWELVEDATA_API_KEY")
    if TWELVEDATA_API_KEY:
        st.sidebar.success("✅ Clé API TwelveData chargée")
    else:
        st.sidebar.error("❌ Clé API manquante")
        st.error("**Configuration requise:** Ajoutez TWELVEDATA_API_KEY dans les secrets Streamlit")
        st.stop()
except Exception as e:
    st.sidebar.error("❌ Erreur de configuration")
    st.error("Vérifiez la configuration des secrets Streamlit")
    st.stop()

# Paires Forex
FOREX_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
    "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

# Fonctions indicateurs techniques
def safe_ema(series, period):
    """EMA sécurisée"""
    try:
        if len(series) < period:
            return pd.Series([np.nan] * len(series), index=series.index)
        return series.ewm(span=period, adjust=False).mean()
    except:
        return pd.Series([np.nan] * len(series), index=series.index)

def safe_rma(series, period):
    """RMA sécurisée (Wilder's MA)"""
    try:
        if len(series) < period:
            return pd.Series([np.nan] * len(series), index=series.index)
        return series.ewm(alpha=1/period, adjust=False).mean()
    except:
        return pd.Series([np.nan] * len(series), index=series.index)

def calculate_hull_ma(close_prices, period=20):
    """Hull Moving Average"""
    try:
        if len(close_prices) < period:
            return pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        
        half_period = max(1, int(period / 2))
        sqrt_period = max(1, int(np.sqrt(period)))
        
        def weighted_ma(data, length):
            if len(data) < length:
                return np.nan
            weights = np.arange(1, length + 1)
            return np.average(data[-length:], weights=weights)
        
        wma1 = close_prices.rolling(window=half_period).apply(
            lambda x: weighted_ma(x.values, len(x)), raw=False
        )
        wma2 = close_prices.rolling(window=period).apply(
            lambda x: weighted_ma(x.values, len(x)), raw=False
        )
        
        raw_hma = 2 * wma1 - wma2
        hma = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: weighted_ma(x.values, len(x)), raw=False
        )
        
        return hma
    except:
        return pd.Series([np.nan] * len(close_prices), index=close_prices.index)

def calculate_rsi(price_data, period=14):
    """RSI Calculation"""
    try:
        if len(price_data) < period + 1:
            return pd.Series([50] * len(price_data), index=price_data.index)
        
        delta = price_data.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = safe_rma(gain, period)
        avg_loss = safe_rma(loss, period)
        
        avg_loss = avg_loss.replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(price_data), index=price_data.index)

def calculate_adx(high, low, close, period=14):
    """ADX Calculation"""
    try:
        if len(high) < period + 1:
            return pd.Series([0] * len(high), index=high.index)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high.diff()
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index
        )
        
        # Smoothed values
        atr = safe_rma(tr, period)
        plus_di = 100 * (safe_rma(plus_dm, period) / atr.replace(0, 1e-10))
        minus_di = 100 * (safe_rma(minus_dm, period) / atr.replace(0, 1e-10))
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
        adx = safe_rma(dx, period)
        
        return adx.fillna(0)
    except:
        return pd.Series([0] * len(high), index=high.index)

def calculate_heiken_ashi(ohlc_df):
    """Heiken Ashi Calculation"""
    try:
        if ohlc_df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        ha_close = (ohlc_df['Open'] + ohlc_df['High'] + ohlc_df['Low'] + ohlc_df['Close']) / 4
        ha_open = pd.Series(index=ohlc_df.index, dtype=float)
        
        ha_open.iloc[0] = (ohlc_df['Open'].iloc[0] + ohlc_df['Close'].iloc[0]) / 2
        
        for i in range(1, len(ohlc_df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        return ha_open, ha_close
    except:
        return pd.Series(dtype=float), pd.Series(dtype=float)

def calculate_smoothed_ha(ohlc_df, period1=10, period2=10):
    """Smoothed Heiken Ashi"""
    try:
        smooth_ohlc = pd.DataFrame({
            'Open': safe_ema(ohlc_df['Open'], period1),
            'High': safe_ema(ohlc_df['High'], period1),
            'Low': safe_ema(ohlc_df['Low'], period1),
            'Close': safe_ema(ohlc_df['Close'], period1)
        })
        
        ha_open, ha_close = calculate_heiken_ashi(smooth_ohlc)
        
        sha_open = safe_ema(ha_open, period2)
        sha_close = safe_ema(ha_close, period2)
        
        return sha_open, sha_close
    except:
        return pd.Series(dtype=float), pd.Series(dtype=float)

def calculate_ichimoku_signal(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """Ichimoku Cloud Signal"""
    try:
        min_length = max(tenkan, kijun, senkou_b)
        if len(high) < min_length:
            return 0
        
        tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
        kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
        
        senkou_a = (tenkan_sen + kijun_sen) / 2
        senkou_b = (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
        
        current_close = close.iloc[-1]
        current_senkou_a = senkou_a.iloc[-1]
        current_senkou_b = senkou_b.iloc[-1]
        
        if pd.isna(current_close) or pd.isna(current_senkou_a) or pd.isna(current_senkou_b):
            return 0
        
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        
        if current_close > cloud_top:
            return 1  # Bullish
        elif current_close < cloud_bottom:
            return -1  # Bearish
        else:
            return 0  # Neutral (inside cloud)
    except:
        return 0

@st.cache_data(ttl=300)
def fetch_twelvedata(symbol, interval='1h', size=300):
    """Récupère les données de TwelveData"""
    if not TWELVEDATA_API_KEY:
        return None
    
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': interval,
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': size,
        'timezone': 'UTC'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'error':
            st.warning(f"API Error pour {symbol}: {data.get('message', 'Unknown error')}")
            return None
        
        if 'values' not in data or not data.get('values'):
            st.warning(f"Pas de données pour {symbol}")
            return None
        
        df = pd.DataFrame(data['values'])
        
        # Conversion des types
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)  # Plus ancien au plus récent
        
        # Renommage
        df.rename(columns={
            'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close'
        }, inplace=True)
        
        # Validation
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        
        if len(df) < 60:
            st.warning(f"Données insuffisantes pour {symbol}: {len(df)} barres")
            return None
        
        return df
        
    except requests.RequestException as e:
        st.error(f"Erreur réseau pour {symbol}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue pour {symbol}: {str(e)}")
        return None

def analyze_signals(data):
    """Analyse tous les signaux techniques"""
    if data is None or len(data) < 60:
        return None
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        open_price = data['Open']
        
        # OHLC4 pour RSI
        ohlc4 = (open_price + high + low + close) / 4
        
        bulls = 0
        bears = 0
        signals = {}
        
        # 1. Hull MA
        try:
            hma = calculate_hull_ma(close, 20)
            if len(hma) >= 2 and not hma.iloc[-2:].isna().any():
                if hma.iloc[-1] > hma.iloc[-2]:
                    bulls += 1
                    signals['HMA'] = "▲"
                elif hma.iloc[-1] < hma.iloc[-2]:
                    bears += 1
                    signals['HMA'] = "▼"
                else:
                    signals['HMA'] = "─"
            else:
                signals['HMA'] = "N/A"
        except:
            signals['HMA'] = "Err"
        
        # 2. RSI
        try:
            rsi = calculate_rsi(ohlc4, 10)
            if not pd.isna(rsi.iloc[-1]):
                rsi_val = rsi.iloc[-1]
                signals['RSI_val'] = f"{rsi_val:.0f}"
                if rsi_val > 50:
                    bulls += 1
                    signals['RSI'] = f"▲({rsi_val:.0f})"
                elif rsi_val < 50:
                    bears += 1
                    signals['RSI'] = f"▼({rsi_val:.0f})"
                else:
                    signals['RSI'] = f"─({rsi_val:.0f})"
            else:
                signals['RSI'] = "N/A"
                signals['RSI_val'] = "N/A"
        except:
            signals['RSI'] = "Err"
            signals['RSI_val'] = "N/A"
        
        # 3. ADX
        try:
            adx = calculate_adx(high, low, close, 14)
            if not pd.isna(adx.iloc[-1]):
                adx_val = adx.iloc[-1]
                signals['ADX_val'] = f"{adx_val:.0f}"
                if adx_val >= 20:
                    bulls += 1
                    bears += 1  # ADX indique la force, pas la direction
                    signals['ADX'] = f"✔({adx_val:.0f})"
                else:
                    signals['ADX'] = f"✖({adx_val:.0f})"
            else:
                signals['ADX'] = "N/A"
                signals['ADX_val'] = "N/A"
        except:
            signals['ADX'] = "Err"
            signals['ADX_val'] = "N/A"
        
        # 4. Heiken Ashi
        try:
            ha_open, ha_close = calculate_heiken_ashi(data)
            if not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
                if ha_close.iloc[-1] > ha_open.iloc[-1]:
                    bulls += 1
                    signals['HA'] = "▲"
                elif ha_close.iloc[-1] < ha_open.iloc[-1]:
                    bears += 1
                    signals['HA'] = "▼"
                else:
                    signals['HA'] = "─"
            else:
                signals['HA'] = "N/A"
        except:
            signals['HA'] = "Err"
        
        # 5. Smoothed Heiken Ashi
        try:
            sha_open, sha_close = calculate_smoothed_ha(data, 10, 10)
            if not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
                if sha_close.iloc[-1] > sha_open.iloc[-1]:
                    bulls += 1
                    signals['SHA'] = "▲"
                elif sha_close.iloc[-1] < sha_open.iloc[-1]:
                    bears += 1
                    signals['SHA'] = "▼"
                else:
                    signals['SHA'] = "─"
            else:
                signals['SHA'] = "N/A"
        except:
            signals['SHA'] = "Err"
        
        # 6. Ichimoku
        try:
            ichi_signal = calculate_ichimoku_signal(high, low, close)
            if ichi_signal == 1:
                bulls += 1
                signals['Ichi'] = "▲"
            elif ichi_signal == -1:
                bears += 1
                signals['Ichi'] = "▼"
            else:
                signals['Ichi'] = "─"
        except:
            signals['Ichi'] = "Err"
        
        # Résultat final
        confluence = max(bulls, bears)
        
        if bulls > bears:
            direction = "HAUSSIER"
        elif bears > bulls:
            direction = "BAISSIER"
        elif bulls == bears and bulls > 0:
            direction = "CONFLIT"
        else:
            direction = "NEUTRE"
        
        return {
            'confluence': confluence,
            'direction': direction,
            'bulls': bulls,
            'bears': bears,
            'rsi_val': signals.get('RSI_val', "N/A"),
            'adx_val': signals.get('ADX_val', "N/A"),
            'signals': signals
        }
        
    except Exception as e:
        print(f"Erreur dans analyze_signals: {e}")
        return None

def get_stars(confluence):
    """Convertit confluence en étoiles"""
    stars_map = {
        6: "⭐⭐⭐⭐⭐⭐",
        5: "⭐⭐⭐⭐⭐", 
        4: "⭐⭐⭐⭐",
        3: "⭐⭐⭐",
        2: "⭐⭐",
        1: "⭐"
    }
    return stars_map.get(confluence, "ATTENDRE")

# Interface utilisateur
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Paramètres")
    
    min_confluence = st.selectbox(
        "Confluence minimum",
        options=[0, 1, 2, 3, 4, 5, 6],
        index=3,
        format_func=lambda x: f"{x} étoile{'s' if x != 1 else ''}"
    )
    
    show_all = st.checkbox("Afficher toutes les paires")
    
    scan_button = st.button(
        "🔍 Lancer le Scan",
        type="primary",
        use_container_width=True,
        disabled=not TWELVEDATA_API_KEY
    )

with col2:
    if scan_button and TWELVEDATA_API_KEY:
        st.info("🔄 Scan en cours...")
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(FOREX_PAIRS)
        
        for i, pair in enumerate(FOREX_PAIRS):
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Analyse: {pair} ({i+1}/{total})")
            
            # Récupération des données
            data = fetch_twelvedata(pair)
            
            if data is not None:
                # Analyse des signaux
                analysis = analyze_signals(data)
                
                if analysis:
                    results.append({
                        'Paire': pair,
                        'Direction': analysis['direction'],
                        'Confluence': analysis['confluence'], 
                        'Étoiles': get_stars(analysis['confluence']),
                        'RSI': analysis['rsi_val'],
                        'ADX': analysis['adx_val'],
                        'Bulls': analysis['bulls'],
                        'Bears': analysis['bears'],
                        'Détails': analysis['signals']
                    })
                else:
                    results.append({
                        'Paire': pair,
                        'Direction': 'ERREUR',
                        'Confluence': 0,
                        'Étoiles': 'N/A',
                        'RSI': 'N/A', 
                        'ADX': 'N/A',
                        'Bulls': 0,
                        'Bears': 0,
                        'Détails': {}
                    })
            else:
                results.append({
                    'Paire': pair,
                    'Direction': 'PAS DE DONNÉES',
                    'Confluence': 0,
                    'Étoiles': 'N/A',
                    'RSI': 'N/A',
                    'ADX': 'N/A', 
                    'Bulls': 0,
                    'Bears': 0,
                    'Détails': {}
                })
            
            # Pause pour respecter les limites API
            if i < total - 1:
                time.sleep(8)
        
        progress_bar.empty()
        status_text.empty()
        
        # Affichage des résultats
        if results:
            df = pd.DataFrame(results)
            
            # Filtrage
            if show_all:
                df_filtered = df.copy()
                st.info(f"📊 Affichage de toutes les {len(df_filtered)} paires")
            else:
                df_filtered = df[df['Confluence'] >= min_confluence].copy()
                st.success(f"🎯 {len(df_filtered)} paire(s) avec confluence >= {min_confluence}")
            
            if not df_filtered.empty:
                # Tri par confluence
                df_sorted = df_filtered.sort_values('Confluence', ascending=False)
                
                # Tableau principal
                display_cols = ['Paire', 'Direction', 'Confluence', 'Étoiles', 'RSI', 'ADX', 'Bulls', 'Bears']
                st.dataframe(df_sorted[display_cols], use_container_width=True, hide_index=True)
                
                # Détails
                with st.expander("📊 Détails des signaux"):
                    for _, row in df_sorted.iterrows():
                        details = row.get('Détails', {})
                        
                        st.write(f"**{row['Paire']}** - {row['Étoiles']} ({row['Confluence']}) - {row['Direction']}")
                        
                        cols = st.columns(6)
                        indicators = ['HMA', 'RSI', 'ADX', 'HA', 'SHA', 'Ichi']
                        
                        for idx, indicator in enumerate(indicators):
                            cols[idx].metric(
                                label=indicator,
                                value=details.get(indicator, "N/A")
                            )
                        
                        st.divider()
            else:
                st.warning("❌ Aucune paire ne correspond aux critères")
        else:
            st.error("❌ Aucun résultat obtenu")

# Informations
with st.expander("ℹ️ Informations"):
    st.markdown("""
    **Signaux analysés:**
    - **HMA(20)**: Hull Moving Average - Tendance
    - **RSI(10)**: Relative Strength Index - Momentum  
    - **ADX(14)**: Average Directional Index - Force de tendance
    - **HA**: Heiken Ashi - Filtrage du bruit
    - **SHA**: Smoothed Heiken Ashi - Tendance lissée
    - **Ichimoku**: Position par rapport au nuage
    
    **Confluence:** Nombre maximum de signaux dans la même direction
    
    **Source:** TwelveData API (données H1)
    """)

st.caption("⚠️ Respectez les limites de l'API TwelveData (120 requêtes/minute)")
 
