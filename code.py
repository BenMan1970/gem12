import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import traceback 
import requests 

st.set_page_config(page_title="Scanner Confluence Forex (TwelveData)", page_icon="⭐", layout="wide")
st.title("🔍 Scanner Confluence Forex Premium (Données Twelve Data)")
st.markdown("*Analyse technique multi-indicateurs avec données TwelveData H1*") # Titre du script Claude

# Gestion de la clé API (inspiré par Claude pour la clarté)
TWELVEDATA_API_KEY = None
try:
    TWELVEDATA_API_KEY = st.secrets.get("TWELVEDATA_API_KEY") # Utiliser .get pour éviter KeyError direct
    if TWELVEDATA_API_KEY:
        st.sidebar.success("✅ Clé API TwelveData chargée")
    else:
        st.sidebar.error("❌ Clé API TwelveData manquante dans les secrets.")
        st.error("**Configuration requise:** Ajoutez TWELVEDATA_API_KEY dans les secrets Streamlit.")
        st.stop() # Arrêter si la clé n'est pas là
except Exception as e:
    st.sidebar.error("❌ Erreur de configuration des secrets.")
    st.error(f"Vérifiez la configuration des secrets Streamlit: {e}")
    st.stop()

FOREX_PAIRS_TWELVEDATA = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", 
    "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP"
]

# --- Fonctions indicateurs techniques (rendues plus "sûres") ---
def safe_ewm_mean(series, **kwargs):
    if series.isnull().all() or len(series.dropna()) < kwargs.get('span', kwargs.get('com', kwargs.get('halflife', 1))): # Vérif plus robuste
        return pd.Series(np.nan, index=series.index)
    return series.ewm(**kwargs).mean()

def ema(s, p): return safe_ewm_mean(s, span=p, adjust=False)
def rma(s, p): return safe_ewm_mean(s, alpha=1/p, adjust=False)

def hull_ma_pine(dc, p=20):
    if len(dc.dropna()) < p + int(np.sqrt(p)): return pd.Series(np.nan, index=dc.index) # Données insuffisantes
    try:
        hl=max(1,int(p/2)); sl=max(1,int(np.sqrt(p)))
        # Assurer que les rolling windows ne sont pas plus grandes que les données
        if len(dc.dropna()) < hl or len(dc.dropna()) < p: return pd.Series(np.nan, index=dc.index)

        wma1=dc.rolling(window=hl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==hl else np.nan, raw=True)
        wma2=dc.rolling(window=p).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==p else np.nan, raw=True)
        
        raw_hma=2*wma1-wma2
        if len(raw_hma.dropna()) < sl : return pd.Series(np.nan, index=dc.index)
        
        hma = raw_hma.rolling(window=sl).apply(lambda x:np.sum(x*np.arange(1,len(x)+1))/np.sum(np.arange(1,len(x)+1)) if len(x)==sl else np.nan, raw=True)
        return hma
    except Exception as e: print(f"Erreur dans hull_ma_pine: {e}"); return pd.Series(np.nan, index=dc.index)

def rsi_pine(po4,p=10):
    if len(po4.dropna()) < p + 1: return pd.Series(50.0, index=po4.index)
    try:
        d=po4.diff();g=d.where(d>0,0.0);l=-d.where(d<0,0.0)
        ag=rma(g,p);al=rma(l,p)
        al_safe = al.replace(0,1e-9) # Eviter division par zéro
        rs=ag/al_safe
        rsi=100-(100/(1+rs));return rsi.fillna(50)
    except Exception as e: print(f"Erreur dans rsi_pine: {e}"); return pd.Series(50.0, index=po4.index)

def adx_pine(h,l,c,p=14):
    if len(h.dropna()) < p*2 or len(l.dropna()) < p*2 or len(c.dropna()) < p*2: # ADX a besoin de plus de données pour se stabiliser
        return pd.Series(0.0, index=h.index)
    try:
        tr1=h-l;tr2=abs(h-c.shift(1));tr3=abs(l-c.shift(1));tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1);atr=rma(tr,p)
        um=h.diff();dm=l.shift(1)-l
        pdm=pd.Series(np.where((um>dm)&(um>0),um,0.0),index=h.index);mdm=pd.Series(np.where((dm>um)&(dm>0),dm,0.0),index=h.index)
        satr=atr.replace(0,1e-9);pdi=100*(rma(pdm,p)/satr);mdi=100*(rma(mdm,p)/satr)
        dxden=(pdi+mdi).replace(0,1e-9);dx=100*(abs(pdi-mdi)/dxden);return rma(dx,p).fillna(0)
    except Exception as e: print(f"Erreur dans adx_pine: {e}"); return pd.Series(0.0, index=h.index)

def heiken_ashi_pine(dfo):
    if len(dfo.dropna()) < 1: return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
    try:
        ha=pd.DataFrame(index=dfo.index)
        ha['HA_Close']=(dfo['Open']+dfo['High']+dfo['Low']+dfo['Close'])/4;ha['HA_Open']=np.nan
        if not dfo.empty: # Devrait toujours être vrai si len > 0
            ha.iloc[0,ha.columns.get_loc('HA_Open')]=(dfo['Open'].iloc[0]+dfo['Close'].iloc[0])/2
            for i in range(1,len(dfo)):ha.iloc[i,ha.columns.get_loc('HA_Open')]=(ha.iloc[i-1,ha.columns.get_loc('HA_Open')]+ha.iloc[i-1,ha.columns.get_loc('HA_Close')])/2
        return ha['HA_Open'],ha['HA_Close']
    except Exception as e: print(f"Erreur dans heiken_ashi_pine: {e}"); return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)

def smoothed_heiken_ashi_pine(dfo,l1=10,l2=10):
    if len(dfo.dropna()) < max(l1,l2) +1 : return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
    try:
        eo=ema(dfo['Open'],l1);eh=ema(dfo['High'],l1);el=ema(dfo['Low'],l1);ec=ema(dfo['Close'],l1)
        hai=pd.DataFrame({'Open':eo,'High':eh,'Low':el,'Close':ec},index=dfo.index).dropna() # Drop NaNs from EMA
        if hai.empty: return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)
            
        hao_i,hac_i=heiken_ashi_pine(hai);
        sho=ema(hao_i,l2);shc=ema(hac_i,l2);return sho,shc
    except Exception as e: print(f"Erreur dans smoothed_heiken_ashi_pine: {e}"); return pd.Series(dtype=float, index=dfo.index), pd.Series(dtype=float, index=dfo.index)

def ichimoku_pine_signal(df_high, df_low, df_close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    min_len_req=max(tenkan_p,kijun_p,senkou_b_p)
    if len(df_high.dropna())<min_len_req or len(df_low.dropna())<min_len_req or len(df_close.dropna())<min_len_req:print(f"Ichi:Data<({len(df_close)}) vs req {min_len_req}.");return 0
    try:
        ts=(df_high.rolling(window=tenkan_p).max()+df_low.rolling(window=tenkan_p).min())/2;ks=(df_high.rolling(window=kijun_p).max()+df_low.rolling(window=kijun_p).min())/2
        sa=(ts+ks)/2;sb=(df_high.rolling(window=senkou_b_p).max()+df_low.rolling(window=senkou_b_p).min())/2
        if pd.isna(df_close.iloc[-1]) or pd.isna(sa.iloc[-1]) or pd.isna(sb.iloc[-1]):print("Ichi:NaN close/spans.");return 0
        ccl=df_close.iloc[-1];cssa=sa.iloc[-1];cssb=sb.iloc[-1];ctn=max(cssa,cssb);cbn=min(cssa,cssb);sig=0
        if ccl>ctn:sig=1
        elif ccl<cbn:sig=-1
        return sig
    except Exception as e: print(f"Erreur dans ichimoku_pine_signal: {e}"); return 0


@st.cache_data(ttl=600) 
def get_data_twelvedata(symbol_td: str, interval_td: str = '1h', outputsize_td: int = 300):
    global TWELVEDATA_API_KEY # Assurez-vous que cette variable globale est bien celle définie en haut
    if not TWELVEDATA_API_KEY: st.error("FATAL: Clé API Twelve Data non chargée."); print("FATAL: Clé API TD non chargée."); return None
    print(f"\n--- Début get_data_twelvedata: sym='{symbol_td}', interval='{interval_td}', outputsize={outputsize_td} ---")
    base_url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol_td, "interval": interval_td, "apikey": TWELVEDATA_API_KEY, "outputsize": outputsize_td, "timezone": "UTC"}
    response = None 
    try:
        response = requests.get(base_url, params=params, timeout=20) # Ajout d'un timeout
        response.raise_for_status() 
        data = response.json()
        print(f"Données brutes TD reçues pour {symbol_td} (status:{data.get('code','N/A')} msg:{data.get('message','N/A')} début): {str(data.get('values',''))[:100]}...")
        if data.get('status') == 'error' or 'values' not in data or not isinstance(data['values'], list) or not data['values']:
            err_msg = data.get('message', f"Pas de données valides ('values') pour {symbol_td}.")
            st.warning(f"Twelve Data: {err_msg} (Intervalle: {interval_td}).")
            print(f"Twelve Data: {err_msg}. Réponse: {data}")
            return None
        df = pd.DataFrame(data['values'])
        if df.empty: st.warning(f"DF vide après parsing JSON pour {symbol_td}."); print(f"DF vide après parsing JSON {symbol_td}."); return None
        
        cols_to_numeric = ['open', 'high', 'low', 'close']; df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True); df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True) # Assurer l'ordre chronologique
        df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
        
        df_ohlc = df[['Open', 'High', 'Low', 'Close']].dropna() # Garder seulement OHLC et supprimer les lignes avec NaN dans ces colonnes
        
        if df_ohlc.empty or len(df_ohlc) < 60: # Seuil minimum pour analyse
            st.warning(f"Données TD insuffisantes pour analyse de {symbol_td} ({len(df_ohlc)} barres valides). Requis: 60.")
            print(f"Données TD insuffisantes pour {symbol_td} ({len(df_ohlc)} barres). Requis: 60.")
            return None
        
        print(f"Données pour {symbol_td} OK. Retour de {len(df_ohlc)} lignes.\n--- Fin get_data_twelvedata {symbol_td} ---\n")
        return df_ohlc
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP TD {symbol_td}: {http_err}"); print(f"ERREUR HTTP TD {symbol_td}:\n{http_err}")
        if response is not None: print(f"Réponse TD: {response.text}"); return None
    except requests.exceptions.RequestException as req_err: # Plus général pour les erreurs réseau
        st.error(f"Erreur Réseau TD {symbol_td}: {req_err}"); print(f"ERREUR RÉSEAU TD {symbol_td}:\n{req_err}"); return None
    except Exception as e:
        st.error(f"Erreur inattendue get_data_td {symbol_td}: {type(e).__name__}"); st.exception(e); print(f"ERREUR INATTENDUE get_data_td {symbol_td}:\n{traceback.format_exc()}"); return None

# --- Fonction calculate_all_signals_pine (Version correctement indentée) ---
def calculate_all_signals_pine(data):
    # ... (la version que je t'ai donnée dans le message précédent, qui est correcte)
    if data is None or len(data) < 60: print(f"calculate_all_signals: Données non fournies ou trop courtes ({len(data) if data is not None else 'None'} lignes)."); return None
    required_cols = ['Open', 'High', 'Low', 'Close']; 
    if not all(col in data.columns for col in required_cols): print(f"calculate_all_signals: Colonnes OHLC manquantes."); return None
    close = data['Close']; high = data['High']; low = data['Low']; open_price = data['Open']
    ohlc4 = (open_price + high + low + close) / 4
    bull_confluences, bear_confluences, signal_details_pine = 0, 0, {}
    try: hma_series = hull_ma_pine(close, 20)
        if len(hma_series) >= 2 and not hma_series.iloc[-2:].isna().any():
            hma_val = hma_series.iloc[-1]; hma_prev = hma_series.iloc[-2]
            if hma_val > hma_prev: bull_confluences += 1; signal_details_pine['HMA'] = "▲"
            elif hma_val < hma_prev: bear_confluences += 1; signal_details_pine['HMA'] = "▼"
            else: signal_details_pine['HMA'] = "─"
        else: signal_details_pine['HMA'] = "N/A"
    except Exception as e: signal_details_pine['HMA'] = "ErrHMA"; print(f"Erreur HMA: {e}")
    try: rsi_series = rsi_pine(ohlc4, 10)
        if len(rsi_series) >=1 and not pd.isna(rsi_series.iloc[-1]):
            rsi_val = rsi_series.iloc[-1]; signal_details_pine['RSI_val'] = f"{rsi_val:.0f}"
            if rsi_val > 50: bull_confluences += 1; signal_details_pine['RSI'] = f"▲({rsi_val:.0f})"
            elif rsi_val < 50: bear_confluences += 1; signal_details_pine['RSI'] = f"▼({rsi_val:.0f})"
            else: signal_details_pine['RSI'] = f"─({rsi_val:.0f})"
        else: signal_details_pine['RSI'] = "N/A"
    except Exception as e: signal_details_pine['RSI'] = "ErrRSI"; signal_details_pine['RSI_val'] = "N/A"; print(f"Erreur RSI: {e}")
    try: adx_series = adx_pine(high, low, close, 14)
        if len(adx_series) >= 1 and not pd.isna(adx_series.iloc[-1]):
            adx_val = adx_series.iloc[-1]; signal_details_pine['ADX_val'] = f"{adx_val:.0f}"
            if adx_val >= 20: bull_confluences += 1; bear_confluences += 1; signal_details_pine['ADX'] = f"✔({adx_val:.0f})"
            else: signal_details_pine['ADX'] = f"✖({adx_val:.0f})"
        else: signal_details_pine['ADX'] = "N/A"
    except Exception as e: signal_details_pine['ADX'] = "ErrADX"; signal_details_pine['ADX_val'] = "N/A"; print(f"Erreur ADX: {e}")
    try: ha_open, ha_close = heiken_ashi_pine(data)
        if len(ha_open) >=1 and len(ha_close) >=1 and not pd.isna(ha_open.iloc[-1]) and not pd.isna(ha_close.iloc[-1]):
            if ha_close.iloc[-1] > ha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['HA'] = "▲"
            elif ha_close.iloc[-1] < ha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['HA'] = "▼"
            else: signal_details_pine['HA'] = "─"
        else: signal_details_pine['HA'] = "N/A"
    except Exception as e: signal_details_pine['HA'] = "ErrHA"; print(f"Erreur HA: {e}")
    try: sha_open, sha_close = smoothed_heiken_ashi_pine(data, 10, 10)
        if len(sha_open) >=1 and len(sha_close) >=1 and not pd.isna(sha_open.iloc[-1]) and not pd.isna(sha_close.iloc[-1]):
            if sha_close.iloc[-1] > sha_open.iloc[-1]: bull_confluences += 1; signal_details_pine['SHA'] = "▲"
            elif sha_close.iloc[-1] < sha_open.iloc[-1]: bear_confluences += 1; signal_details_pine['SHA'] = "▼"
            else: signal_details_pine['SHA'] = "─"
        else: signal_details_pine['SHA'] = "N/A"
    except Exception as e: signal_details_pine['SHA'] = "ErrSHA"; print(f"Erreur SHA: {e}")
    try: ichimoku_signal_val = ichimoku_pine_signal(high, low, close)
        if ichimoku_signal_val == 1: bull_confluences += 1; signal_details_pine['Ichi'] = "▲"
        elif ichimoku_signal_val == -1: bear_confluences += 1; signal_details_pine['Ichi'] = "▼"
        elif ichimoku_signal_val == 0 and (len(data) < max(9,26,52) or (len(data) > 0 and pd.isna(data['Close'].iloc[-1]))): signal_details_pine['Ichi'] = "N/D"
        else: signal_details_pine['Ichi'] = "─"
    except Exception as e: signal_details_pine['Ichi'] = "ErrIchi"; print(f"Erreur Ichi: {e}")
    confluence_value=max(bull_confluences,bear_confluences); direction="NEUTRE"
    if bull_confluences > bear_confluences: direction="HAUSSIER"
    elif bear_confluences > bull_confluences: direction="BAISSIER"
    elif bull_confluences == bear_confluences and bull_confluences > 0: direction="CONFLIT"
    return{'confluence_P':confluence_value,'direction_P':direction,'bull_P':bull_confluences,'bear_P':bear_confluences,'rsi_P':signal_details_pine.get('RSI_val',"N/A"),'adx_P':signal_details_pine.get('ADX_val',"N/A"),'signals_P':signal_details_pine}

def get_stars_pine(confluence_value):
    if confluence_value == 6: return "⭐⭐⭐⭐⭐⭐"; elif confluence_value == 5: return "⭐⭐⭐⭐⭐"; 
    elif confluence_value == 4: return "⭐⭐⭐⭐"; elif confluence_value == 3: return "⭐⭐⭐"; 
    elif confluence_value == 2: return "⭐⭐"; elif confluence_value == 1: return "⭐"; else: return "WAIT"

col1,col2=st.columns([1,3])
with col1: # Paramètres
    st.subheader("⚙️ Paramètres");min_conf=st.selectbox("Confluence minimum",(0,1,2,3,4,5,6),index=3,format_func=lambda x:f"{x} étoiles")
    show_all=st.checkbox("Afficher toutes les paires", True); scan_dis_td=TWELVEDATA_API_KEY is None; 
    scan_tip_td="Clé TwelveData non configurée." if scan_dis_td else "Scanner les paires Forex"
    scan_btn=st.button("🔍 Lancer le Scan",type="primary",use_container_width=True,disabled=scan_dis_td,help=scan_tip_td)
with col2: # Résultats
    if scan_btn and TWELVEDATA_API_KEY: # S'assurer que la clé est là avant de scanner
        st.info("🔄 Scan en cours..."); pr_res=[]; pb=st.progress(0); stx=st.empty()
        total_pairs = len(FOREX_PAIRS_TWELVEDATA)
        for i,pair_symbol in enumerate(FOREX_PAIRS_TWELVEDATA): 
            pnd=pair_symbol; cp=(i+1)/total_pairs; pb.progress(cp); stx.text(f"Analyse: {pnd} ({i+1}/{total_pairs})")
            d_h1_td=get_data_twelvedata(pair_symbol,interval_td="1h",outputsize_td=300) 
            if d_h1_td is not None and not d_h1_td.empty:
                sigs=calculate_all_signals_pine(d_h1_td)
                if sigs:strs=get_stars_pine(sigs['confluence_P']);rd={'Paire':pnd,'Direction':sigs['direction_P'],'Conf. (0-6)':sigs['confluence_P'],'Étoiles':strs,'RSI':sigs['rsi_P'],'ADX':sigs['adx_P'],'Bull':sigs['bull_P'],'Bear':sigs['bear_P'],'details':sigs['signals_P']};pr_res.append(rd)
                else:pr_res.append({'Paire':pnd,'Direction':'ERR CALC','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Calcul échoué (TD)'}})
            else:pr_res.append({'Paire':pnd,'Direction':'ERR DATA','Conf. (0-6)':0,'Étoiles':'N/A','RSI':'N/A','ADX':'N/A','Bull':0,'Bear':0,'details':{'Info':'Données TD non dispo (logs)'}})
            if i < total_pairs - 1 : print(f"Pause 8s pour limite TD..."); time.sleep(8) 
        pb.empty();stx.empty()
        if pr_res:
            dfa=pd.DataFrame(pr_res);dfd=dfa[dfa['Conf. (0-6)']>=min_conf].copy()if not show_all else dfa.copy()
            if not show_all:st.success(f"🎯 {len(dfd)} paire(s) avec {min_conf}+ confluence (TD).")
            else:st.info(f"🔍 Affichage {len(dfd)} paires (TD).")
            if not dfd.empty:
                dfds=dfd.sort_values('Conf. (0-6)',ascending=False);vcs=[c for c in['Paire','Direction','Conf. (0-6)','Étoiles','RSI','ADX','Bull','Bear']if c in dfds.columns]
                st.dataframe(dfds[vcs],use_container_width=True,hide_index=True)
                with st.expander("📊 Détails des signaux (TD)"):
                    for _,r in dfds.iterrows():
                        sm=r.get('details',{});
                        if not isinstance(sm,dict):sm={'Info':'Détails non dispo'}
                        st.write(f"**{r.get('Paire','N/A')}** - {r.get('Étoiles','N/A')} ({r.get('Conf. (0-6)','N/A')}) - Dir: {r.get('Direction','N/A')}")
                        dc=st.columns(6);so=['HMA','RSI','ADX','HA','SHA','Ichi']
                        for idx,sk in enumerate(so):dc[idx].metric(label=sk,value=sm.get(sk,"N/P"))
                        st.divider()
            else:st.warning(f"❌ Aucune paire avec critères (TD). Vérifiez erreurs.")
        else:st.error("❌ Aucune paire traitée (TD). Vérifiez logs serveur.")
with st.expander("ℹ️ Informations"):
    st.markdown("""**Signaux:** HMA(20),RSI(10),ADX(14)>=20,HA,SHA(10,10),Ichi(9,26,52). **Source:** TwelveData API.""")
st.caption("Scanner H1 (TwelveData). Respectez limites API.")
   
