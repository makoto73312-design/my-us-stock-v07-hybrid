import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import re
import traceback
from datetime import datetime

# --- 1. 網頁核心外觀配置 ---
st.set_page_config(page_title="🔮 美股量化沙盒 V07.1", page_icon="🔮", layout="wide")
st.title("🔮 美股量化投資沙盒 V07.1 (完整功能與成效驗證升級版)")
st.caption("🚀 已實裝 **四大專屬分頁 + 布林 6 大專家分類 + 昨日買訊成效驗證 + 雲端自選管理**。")

# --- 2. 側邊欄控制台 ---
st.sidebar.header("⚙️ 全自動大掃描設定")

GSHEET_URL = "https://docs.google.com/spreadsheets/d/1491qc1Y59PwCOWaPZblpAieR0_iCI-7KKLtZUuG7Qe4/edit?usp=sharing"

@st.cache_data(ttl=60)
def get_tickers_from_sheet(url):
    try:
        if "docs.google.com" not in url:
            return "NVDA, AAPL, TSLA, MSFT, AMD", {}
        csv_url = url.split("/edit")[0] + "/export?format=csv"
        df = pd.read_csv(csv_url, header=None)
        tickers = df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
        custom_names_dict = {}
        valid_tickers = [t for t in tickers if not any(c >= '\u4e00' and c <= '\u9fff' for c in t) and len(t) > 0 and t != "股票代號"]
        ticker_str = ", ".join(valid_tickers) if valid_tickers else "NVDA, AAPL, TSLA, MSFT, AMD"
        return ticker_str, custom_names_dict
    except Exception:
        return "NVDA, AAPL, TSLA, MSFT, AMD", {}

default_tickers, cloud_names_dict = get_tickers_from_sheet(GSHEET_URL)

with st.sidebar.expander("🌐 雲端自選清單管理", expanded=False):
    st.markdown(f"[🔗 點此開啟 Google 雲端試算表]({GSHEET_URL})")
    if st.button("🔄 強制刷新雲端清單快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"已同步雲端標的數：{len(cloud_names_dict)} 筆中文名稱對照")

tickers_input = st.sidebar.text_area("📡 當前雲端同步清單", default_tickers, height=120)

temp_raw_list = [t.strip().upper() for t in re.split(r'[\n\r,\s]+', tickers_input) if t.strip()]
ticker_list = list(dict.fromkeys(temp_raw_list))

if ticker_list:
    if "us_debug_tk" not in st.session_state or st.session_state["us_debug_tk"] not in ticker_list:
        st.session_state["us_debug_tk"] = ticker_list[0]
    if "us_7d_tk" not in st.session_state or st.session_state["us_7d_tk"] not in ticker_list:
        st.session_state["us_7d_tk"] = ticker_list[0]
    if "us_debug_st" not in st.session_state:
        st.session_state["us_debug_st"] = "A: 激進動能型"

backtest_days = st.sidebar.slider("歷史回測天數設定", min_value=100, max_value=500, value=300, step=50)
enable_fcf_filter = st.sidebar.checkbox("🛡️ 啟用「FCF 負值」強制攔截", value=True)

st.sidebar.divider()
show_debug_log = st.sidebar.checkbox("🐛 顯示系統診斷日誌", value=False)

def clean_and_flatten_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        found_level = None
        for level in range(df.columns.nlevels):
            level_vals = [str(c).title() for c in df.columns.get_level_values(level)]
            if 'Close' in level_vals:
                found_level = level
                break
        if found_level is not None:
            df.columns = df.columns.get_level_values(found_level)
        else:
            df.columns = df.columns.get_level_values(-1)
            
    standard_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close'
    }
    new_cols = []
    for c in df.columns:
        c_str = str(c)
        if c_str.lower() in standard_map:
            new_cols.append(standard_map[c_str.lower()])
        else:
            new_cols.append(c_str)
    df.columns = new_cols
    return df

def extract_stock_from_chunk(df_chunk, ticker):
    if df_chunk is None or df_chunk.empty:
        return pd.DataFrame()
    if not isinstance(df_chunk.columns, pd.MultiIndex):
        return clean_and_flatten_df(df_chunk)
    for lvl in range(df_chunk.columns.nlevels):
        if ticker in df_chunk.columns.get_level_values(lvl):
            try:
                df_sub = df_chunk.xs(ticker, level=lvl, axis=1).copy()
                df_sub = clean_and_flatten_df(df_sub)
                if 'Close' in df_sub.columns and not df_sub.dropna(subset=['Close']).empty:
                    return df_sub.dropna(subset=['Close'])
            except Exception:
                pass
    return pd.DataFrame()

# --- 3. 🌐 V07 美股大環境與總經雷達 ---
@st.cache_data(ttl=1800)
def fetch_us_macro_dataframe():
    try:
        macro_raw = yf.download(["^VIX", "SPY"], period="2y", progress=False)
        vix_df = extract_stock_from_chunk(macro_raw, '^VIX')
        spy_df = extract_stock_from_chunk(macro_raw, 'SPY')
        
        vix_c = vix_df[['Close']].rename(columns={'Close': 'VIX'})
        spy_c = spy_df[['Close']].rename(columns={'Close': 'SPY_Close'})
        
        vix_c.index = pd.to_datetime(pd.to_datetime(vix_c.index).date)
        spy_c.index = pd.to_datetime(pd.to_datetime(spy_c.index).date)

        spy_c['SPY_MA200'] = spy_c['SPY_Close'].rolling(200).mean().fillna(spy_c['SPY_Close'])
        spy_c['Market_Bull'] = spy_c['SPY_Close'] >= spy_c['SPY_MA200']
        
        df_macro = spy_c.join(vix_c, how='left').ffill().bfill().dropna()
        
        latest_vix = float(df_macro['VIX'].iloc[-1])
        latest_bull = bool(df_macro['Market_Bull'].iloc[-1])
        if latest_vix >= 25 or not latest_bull:
            posture_auto = "🥶 極度謹慎型 (大盤空頭/高恐慌)"
        elif latest_vix <= 15 and latest_bull:
            posture_auto = "🚀 大膽進攻型 (晴天多頭行情)"
        else:
            posture_auto = "🛡️ 標準平衡型 (常態橫盤整理)"
            
        return df_macro, latest_vix, latest_bull, posture_auto, "SUCCESS"
    except Exception as e:
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=500, freq='D')
        df_macro = pd.DataFrame({'VIX': 18.0, 'Market_Bull': True, 'SPY_Close': 500.0}, index=dates)
        return df_macro, 18.0, True, "🛡️ 標準平衡型 (預設)", f"ERROR: {str(e)}"

df_macro, vix_score, is_spy_bull, market_posture, macro_status = fetch_us_macro_dataframe()

# --- 4. 🏢 V07 基本面雷達 ---
@st.cache_data(ttl=3600)
def fetch_fundamental_and_news(ticker, cloud_dict):
    f_info = {
        "sector_tag": "🇺🇸 美股企業", "pe": "-", "fcf": "-", "rev_growth": "-", 
        "fcf_status": "UNKNOWN", "near_earnings": False, "quality_tag": "一般"
    }
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        pe = info.get("trailingPE", None)
        fcf = info.get("freeCashflow", None)
        rev_g = info.get("revenueGrowth", None)
        
        if pe is not None: f_info["pe"] = f"{pe:.1f}倍"
        if fcf is not None:
            f_info["fcf"] = f"${fcf / 1e8:.1f}億"
            if fcf < 0: f_info["fcf_status"] = "NEGATIVE"
            else: f_info["fcf_status"] = "POSITIVE"
        else:
            f_info["fcf_status"] = "UNKNOWN"

        if rev_g is not None: f_info["rev_growth"] = f"{rev_g * 100:+.1f}%"
        if (fcf is not None and fcf > 0) and (rev_g is not None and rev_g > 0.15):
            f_info["quality_tag"] = "🔥 財報雙強"
    except Exception:
        pass
    return f_info

# --- 5. 技術指標與布林通道計算 ---
def calculate_indicators(df):
    df = clean_and_flatten_df(df)
    high_low_diff = (df['High'] - df['Low']).replace(0, 0.001) 
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff
    df['價量動能流'] = (df['Volume'] * mf_multiplier / 1000000).round(2)
    df['CLV'] = (df['Close'] - df['Low']) / high_low_diff
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean().fillna(df['Close'] * 0.03)

    std20 = df['Close'].rolling(20).std().fillna(df['Close'] * 0.02)
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_Mid'] = df['MA20']
    df['BB_Upper'] = df['BB_Mid'] + (2.0 * std20)
    df['BB_Lower'] = df['BB_Mid'] - (2.0 * std20)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'].replace(0, 1.0)
    df['BB_Squeeze'] = df['BB_Width'] <= df['BB_Width'].rolling(100, min_periods=20).quantile(0.25)

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['50MA'] = df['Close'].rolling(50).mean()
    df['200MA'] = df['Close'].rolling(200).mean()
    df['ROC14'] = df['Close'].pct_change(14)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / ema_loss.replace(0, 0.001)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['動能流_Q80'] = df['價量動能流'].rolling(50).quantile(0.8)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    macd_shrink = [0] * len(df)
    hist = df['MACD_Hist'].values
    for i in range(1, len(df)):
        if hist[i] < 0 and hist[i] > hist[i-1]: macd_shrink[i] = macd_shrink[i-1] + 1
        else: macd_shrink[i] = 0
    df['MACD_Shrink'] = macd_shrink
    return df

# --- 6. V07.1 回測與成效驗證計算 ---
def run_backtest_engine_v07_us(df_stock, df_macro_input, strategy_name, days, fund_info, 
                               fee_rate=0.0005, tax_rate=0.0, slippage=0.001):
    df_st = clean_and_flatten_df(df_stock.copy())
    df_st.index = pd.to_datetime(pd.to_datetime(df_st.index).date)
    
    valid_df = df_st.join(df_macro_input[['VIX', 'Market_Bull', 'SPY_Close']], how='left').ffill().bfill().dropna().tail(days + 1).copy()
    
    if len(valid_df) < 10:
        return ("⚠️ 數據不足", 0.0, 0.0, 0, "0.00", "D級", "🛑 數據不足", "-", "-", "-", 
                [], [], [], valid_df, 0.0, 0.0, "0.0%", "0.0%", "0.0%", "0.0%", "0.00", "0.0%", "0.0%", "0/7 (無)", [], "常態通道內", 0, {})

    valid_df['Stock_Ret20'] = valid_df['Close'].pct_change(20)
    valid_df['Macro_Ret20'] = valid_df['SPY_Close'].pct_change(20)
    valid_df['RS_20'] = valid_df['Stock_Ret20'] - valid_df['Macro_Ret20']

    capital = 1.0
    equity_curve = [1.0]
    has_position = False
    entry_price, entry_price_with_cost, current_stop_price, highest_price_prior = 0.0, 0.0, 0.0, 0.0
    
    pos_min_low, pos_max_high = 0.0, 0.0
    mae_list, mfe_list = [], []
    win_returns, loss_returns, all_returns = [], [], []

    trade_logs, plot_buys, plot_sells = [], [], []

    dates = valid_df.index
    opens, highs, lows, closes = valid_df['Open'].values, valid_df['High'].values, valid_df['Low'].values, valid_df['Close'].values
    vixs, m_bulls = valid_df['VIX'].values, valid_df['Market_Bull'].values

    s_ma_vals = valid_df['MA5'].values if "A:" in strategy_name else (valid_df['MA14'].values if "B:" in strategy_name else valid_df['MA20'].values)
    m50_vals, m200_vals = valid_df['50MA'].values, valid_df['200MA'].values
    r14_vals, rsi_vals = valid_df['ROC14'].values, valid_df['RSI_14'].values
    vol_vals, vol_m20_vals = valid_df['Volume'].values, valid_df['Vol_MA20'].values
    m_shrink_vals, m_hist_vals = valid_df['MACD_Shrink'].values, valid_df['MACD_Hist'].values
    clv_vals, atr_vals = valid_df['CLV'].values, valid_df['ATR14'].values
    
    # 🛠️ 核心修復：補充補齊 valid_df['BB_Lower'].values 的解包變數
    bb_mid_vals, bb_upper_vals, bb_lower_vals, bb_sqz_vals = valid_df['BB_Mid'].values, valid_df['BB_Upper'].values, valid_df['BB_Lower'].values, valid_df['BB_Squeeze'].values
    
    pv_flow_vals, q80_vals = valid_df['價量動能流'].values, valid_df['動能流_Q80'].values
    rs_vals = valid_df['RS_20'].values

    pending_buy_signal = False
    signal_yesterday_triggered = False
    last_exit_was_today = False

    for i in range(1, len(valid_df)):
        date_str = dates[i].strftime('%Y-%m-%d')
        open_p, high_p, low_p, close_p = opens[i], highs[i], lows[i], closes[i]
        
        vix_y, bull_y = vixs[i-1], m_bulls[i-1]
        if vix_y >= 25 or not bull_y: rsi_max, vol_mult, dip_pct = 65, 1.50, -0.15
        elif vix_y <= 15 and bull_y: rsi_max, vol_mult, dip_pct = 75, 1.05, -0.08
        else: rsi_max, vol_mult, dip_pct = 70, 1.20, -0.10

        atr_p = atr_vals[i-1]
        atr_multiplier = 2.0 if "C:" in strategy_name else 1.5

        if not has_position:
            if pending_buy_signal:
                has_position = True
                pending_buy_signal = False
                entry_price = open_p * (1 + slippage)
                entry_price_with_cost = entry_price * (1 + fee_rate)
                highest_price_prior = open_p
                current_stop_price = entry_price - (atr_multiplier * atr_p)
                pos_min_low, pos_max_high = low_p, high_p
                
                trade_logs.append({"交易日期": date_str, "動作狀態": "🟢 買入進場 (BUY)", "執行價格": f"${entry_price:.2f}", "單筆報酬": "-"})
                plot_buys.append((dates[i], entry_price))
        else:
            pos_min_low = min(pos_min_low, low_p)
            pos_max_high = max(pos_max_high, high_p)

            new_trailing_stop = highest_price_prior - (atr_multiplier * atr_p)
            current_stop_price = max(current_stop_price, new_trailing_stop)
            
            is_exit = False
            exit_price = 0.0

            if low_p <= current_stop_price:
                is_exit = True
                exit_price = min(open_p, current_stop_price) * (1 - slippage)

            if is_exit:
                exit_price_after_cost = exit_price * (1 - fee_rate - tax_rate)
                trade_return = (exit_price_after_cost - entry_price_with_cost) / entry_price_with_cost
                
                capital *= (1 + trade_return)
                all_returns.append(trade_return)

                mae_val = (pos_min_low - entry_price) / entry_price
                mfe_val = (pos_max_high - entry_price) / entry_price
                mae_list.append(mae_val)
                mfe_list.append(mfe_val)

                if trade_return > 0: win_returns.append(trade_return)
                else: loss_returns.append(abs(trade_return))

                has_position = False
                if i == len(valid_df) - 1:
                    last_exit_was_today = True

                trade_logs.append({"交易日期": date_str, "動作狀態": "🔴 賣出出場 (SELL)", "執行價格": f"${exit_price:.2f}", "單筆報酬": f"{trade_return*100:+.2f}%"})
                plot_sells.append((dates[i], exit_price))

        equity_curve.append(capital)

        if has_position:
            highest_price_prior = max(highest_price_prior, high_p)
        else:
            c_p, sma_p, m50_p, m200_p = closes[i], s_ma_vals[i], m50_vals[i], m200_vals[i]
            r14_p, rsi_p, clv_p = r14_vals[i], rsi_vals[i], clv_vals[i]
            vol_p, vol_m20_p = vol_vals[i], vol_m20_vals[i]
            m_shrink_p, m_hist_p = m_shrink_vals[i], m_hist_vals[i]
            m_hist_y = m_hist_vals[i-1]
            pv_flow_p, q80_p, rs_p = pv_flow_vals[i], q80_vals[i], rs_vals[i]
            bb_upper_p, bb_sqz_y = bb_upper_vals[i], bb_sqz_vals[i-1]
            m50_y = m50_vals[i-3] if i >= 3 else m50_p

            if "A:" in strategy_name:
                if (m_shrink_p >= 1 or (m_hist_p > m_hist_y and m_hist_p > 0)) and r14_p > 0 and rsi_p < rsi_max: 
                    pending_buy_signal = True
            elif "B:" in strategy_name:
                if c_p > sma_p and vol_p > vol_m20_p * vol_mult and clv_p >= 0.65 and rs_p > 0 and (bb_sqz_y or c_p >= bb_upper_p * 0.98): 
                    pending_buy_signal = True
            elif "C:" in strategy_name:
                if c_p > bb_upper_p and vol_p > vol_m20_p * (vol_mult * 1.1) and clv_p >= 0.70 and rs_p > 0.02: 
                    pending_buy_signal = True
            elif "D:" in strategy_name:
                if m200_p > 0 and (c_p - m200_p)/m200_p <= dip_pct and rsi_p < 35 and m_shrink_p >= 1 and c_p > opens[i]: 
                    pending_buy_signal = True
            elif "E:" in strategy_name:
                if pv_flow_p > q80_p and pv_flow_p > 0 and c_p > m50_p and m50_p >= m50_y and rs_p > 0: 
                    pending_buy_signal = True

            if i == len(valid_df) - 2 and pending_buy_signal:
                signal_yesterday_triggered = True

    today_open_p, today_close_p = opens[-1], closes[-1]
    today_intraday_ret = (today_close_p - today_open_p) / today_open_p
    yesterday_verification = {
        "yesterday_buy": signal_yesterday_triggered,
        "today_open": f"${today_open_p:.2f}",
        "today_close": f"${today_close_p:.2f}",
        "today_ret_pct": f"{today_intraday_ret * 100:+.2f}%",
        "raw_today_ret": today_intraday_ret,
        "is_win": "🟢 獲利" if today_intraday_ret > 0 else "🔴 虧損"
    }

    latest_idx = -1
    d1_bull = bool(m_bulls[latest_idx])
    d2_vix = bool(vixs[latest_idx] < 22.0)
    d3_rsi = bool(45.0 <= rsi_vals[latest_idx] <= 75.0)
    d4_vol = bool(vol_vals[latest_idx] > vol_m20_vals[latest_idx])
    d5_macd = bool(m_hist_vals[latest_idx] > 0 or m_shrink_vals[latest_idx] >= 1)
    d6_fcf = bool(fund_info.get("fcf_status") != "NEGATIVE")
    d7_rs = bool(rs_vals[latest_idx] > 0.0)

    matrix_7d_details = [
        {"戰術維度項目": "1. 大盤位階 (200MA)", "檢核標準": "指數高於年線 (多頭市場)", "當前狀態": "✅ 符合" if d1_bull else "❌ 未達標"},
        {"戰術維度項目": "2. VIX 恐慌位階", "檢核標準": "恐慌指數 < 22 (低風險)", "當前狀態": "✅ 符合" if d2_vix else "❌ 高恐慌"},
        {"戰術維度項目": "3. RSI 區間動能", "檢核標準": "14日 RSI 介於 45~75 (健康升勢)", "當前狀態": "✅ 符合" if d3_rsi else "❌ 過熱/過冷"},
        {"戰術維度項目": "4. 攻擊量能發動", "檢核標準": "當日成交量 > 20日均量", "當前狀態": "✅ 符合" if d4_vol else "❌ 量能平淡"},
        {"戰術維度項目": "5. MACD 柱狀動能", "檢核標準": "MACD 柱狀體翻紅或綠柱連續收斂", "當前狀態": "✅ 符合" if d5_macd else "❌ 柱體弱化"},
        {"戰術維度項目": "6. 自由現金流 FCF", "檢核標準": "近四季 FCF >= 0 (營運健全)", "當前狀態": "✅ 符合" if d6_fcf else "❌ 現金流赤字"},
        {"戰術維度項目": "7. 相對強弱 RS20", "檢核標準": "近 20 日漲幅跑贏大盤 Alpha > 0", "當前狀態": "✅ 符合" if d7_rs else "❌ 跑輸大盤"}
    ]

    score_7d = sum([d1_bull, d2_vix, d3_rsi, d4_vol, d5_macd, d6_fcf, d7_rs])
    tag_7d = "極強" if score_7d >= 6 else ("強勢" if score_7d >= 4 else ("中性" if score_7d >= 3 else "偏弱"))
    matrix_7d_str = f"{score_7d}/7 ({tag_7d})"

    last_c, last_l = closes[latest_idx], lows[latest_idx]
    last_mid, last_up, last_low = bb_mid_vals[latest_idx], bb_upper_vals[latest_idx], bb_lower_vals[latest_idx]
    last_sqz = bb_sqz_vals[latest_idx]

    if last_sqz:
        bb_status_str = "🔥 帶狀極致壓縮 (準備發動)"
    elif last_c >= last_up:
        bb_status_str = "🚀 突破布林上軌 (強勢多頭)"
    elif last_l <= last_low:
        bb_status_str = "💎 觸及布林下軌 (超賣回歸)"
    elif last_c < last_mid:
        bb_status_str = "⚠️ 跌破 20MA 中軌 (離場防守)"
    elif abs(last_c - last_mid) / last_mid <= 0.015:
        bb_status_str = "🛡️ 貼近 20MA 中軌 (回檔支撐)"
    else:
        bb_status_str = "⚖️ 常態通道內整理"

    total_trades = len(all_returns)
    win_trades = len(win_returns)
    total_return = capital - 1.0
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    avg_win = np.mean(win_returns) if win_trades > 0 else 0.0
    avg_loss = np.mean(loss_returns) if (total_trades - win_trades) > 0 else 0.0
    
    expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
    
    gross_profit = np.sum(win_returns)
    gross_loss = np.sum(loss_returns)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
    pf_str = "無限" if profit_factor == 99.9 else f"{profit_factor:.2f}"

    years = max(len(valid_df) / 252.0, 0.1)
    cagr = (capital ** (1.0 / years)) - 1.0 if capital > 0 else -1.0

    eq_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / peaks
    mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    daily_returns = pd.Series(eq_arr).pct_change().dropna()
    std_ret = daily_returns.std()
    sharpe = (daily_returns.mean() / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    avg_mae = np.mean(mae_list) if len(mae_list) > 0 else 0.0
    avg_mfe = np.mean(mfe_list) if len(mfe_list) > 0 else 0.0

    if expectancy > 0.02 and sharpe > 1.0 and abs(mdd) < 0.15: grade = "S級 (極優)"
    elif expectancy > 0.01 and sharpe > 0.5: grade = "A級 (優良)"
    elif expectancy > 0.0 and total_return > 0: grade = "B級 (標準)"
    elif total_return > -0.05: grade = "C級 (平庸)"
    else: grade = "D級 (劣等)"

    current_close = closes[-1]
    if has_position:
        current_status = "📦 獲利續抱中 (HOLD)"
        unrealized_pnl = (current_close - entry_price_with_cost) / entry_price_with_cost
        pnl_str = f"{unrealized_pnl*100:+.2f}%"
        entry_price_str = f"${entry_price:.2f}"
        sl_price_str = f"${current_stop_price:.2f}"
    elif pending_buy_signal:
        current_status = "🟢 買入訊號/新進場 (BUY)"
        entry_price_str = f"${current_close:.2f}"
        sl_price_str = f"${current_close - (atr_multiplier * atr_vals[-1]):.2f}"
        pnl_str = "0.00%"
    elif last_exit_was_today:
        current_status = "🔴 觸發防守賣出 (SELL)"
        entry_price_str = "-"
        sl_price_str = "-"
        pnl_str = "-"
    else:
        current_status = "💵 空手觀望 (CASH)"
        entry_price_str = "-"
        sl_price_str = "-"
        pnl_str = "-"

    if enable_fcf_filter and fund_info["fcf_status"] == "NEGATIVE" and ("HOLD" in current_status or "BUY" in current_status):
        current_status = "⚠️ 現金流赤字/風控阻擋 (CASH)"

    latest_rs = rs_vals[-1] * 100 if len(rs_vals) > 0 else 0.0
    rs_tag = f"{latest_rs:+.1f}%"

    return ("📡 運算完畢", total_return, win_rate, total_trades, pf_str, grade, 
            current_status, entry_price_str, sl_price_str, pnl_str, trade_logs, 
            plot_buys, plot_sells, valid_df, entry_price, current_stop_price, rs_tag,
            f"{expectancy*100:+.2f}%", f"{cagr*100:+.1f}%", f"{mdd*100:.1f}%", 
            f"{sharpe:.2f}", f"{avg_mae*100:.1f}%", f"{avg_mfe*100:+.1f}%", 
            matrix_7d_str, matrix_7d_details, bb_status_str, score_7d, yesterday_verification)

# ⚡ 單個股票處理 Worker
def process_single_stock_us(ticker, df_stock, cloud_dict, backtest_days, df_macro_data, strategies):
    try:
        if df_stock is None or df_stock.empty or len(df_stock) < 10:
            stock_reports = []
            for strat in strategies:
                stock_reports.append({
                    "股票代號": ticker, "當前市價": "-", "策略手法": strat,
                    "倉位狀態": "🛑 數據不足", "期望值 Expectancy": "0.0%", "七維戰術矩陣": "0/7 (無)", "布林通道位階": "數據不足",
                    "綜合評級": "D級", "大盤 Alpha (RS20)": "0.0%", "年化 CAGR": "0.0%", 
                    "最大回撤 MDD": "0.0%", "夏普比率 Sharpe": "0.00", "平均浮虧 MAE": "0.0%", 
                    "平均浮盈 MFE": "0.0%", "建議進場價": "-", "未實現損益": "-", 
                    "ATR動態防守價": "-", "複利總報酬": "0.0%", 
                    "歷史勝率": "0.0%", "交易次數": 0, "獲利因子": "0.00", "7D得分": 0, "昨日買訊": False
                })
            return stock_reports, {}, f"❌ [{ticker}] 批次資料中找不到 K 線數據"

        df_stock = clean_and_flatten_df(df_stock)
        df_stock = calculate_indicators(df_stock)
        
        df_temp_clean = df_stock.dropna(subset=['Close'])
        current_close = float(df_temp_clean['Close'].iloc[-1]) if not df_temp_clean.empty else 0.0
        fund_info = fetch_fundamental_and_news(ticker, cloud_dict)

        stock_reports = []
        stock_details = {}
        
        for strat in strategies:
            (radar, ret, win, trades, pf, grade, cur_status, entry_price_val, 
             sl_price, pnl, t_logs, p_buys, p_sells, v_df, raw_entry, raw_sl, 
             rs_tag, expectancy_str, cagr_str, mdd_str, sharpe_str, mae_str, mfe_str, 
             matrix_7d_str, matrix_7d_details, bb_status_str, score_7d_num, yest_ver) = run_backtest_engine_v07_us(
                df_stock, df_macro_data, strat, backtest_days, fund_info
            )
            
            stock_details[(ticker, strat)] = {
                "logs": pd.DataFrame(t_logs), "buys": p_buys, "sells": p_sells, 
                "v_df": v_df, "matrix_7d_details": matrix_7d_details, "matrix_7d_str": matrix_7d_str,
                "yest_ver": yest_ver
            }

            stock_reports.append({
                "股票代號": ticker, "當前市價": f"${current_close:.2f}", "策略手法": strat,
                "倉位狀態": cur_status, "期望值 Expectancy": expectancy_str,
                "七維戰術矩陣": matrix_7d_str, "布林通道位階": bb_status_str, "綜合評級": grade, "大盤 Alpha (RS20)": rs_tag, 
                "年化 CAGR": cagr_str, "最大回撤 MDD": mdd_str, "夏普比率 Sharpe": sharpe_str, 
                "平均浮虧 MAE": mae_str, "平均浮盈 MFE": mfe_str, "建議進場價": entry_price_val, 
                "未實現損益": pnl, "ATR動態防守價": sl_price, "複利總報酬": f"{ret * 100:+.2f}%", 
                "歷史勝率": f"{win * 100:.1f}%", "交易次數": trades, "獲利因子": pf, "7D得分": score_7d_num,
                "昨日買訊": yest_ver.get("yesterday_buy", False),
                "今日開盤": yest_ver.get("today_open", "-"),
                "今日收盤": yest_ver.get("today_close", "-"),
                "今日實質漲跌": yest_ver.get("today_ret_pct", "-"),
                "當日驗證": yest_ver.get("is_win", "-")
            })
        return stock_reports, stock_details, "SUCCESS"
    except Exception as e:
        err_detail = f"💥 [{ticker}] 運算例外: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        stock_reports = []
        for strat in strategies:
            stock_reports.append({
                "股票代號": ticker, "當前市價": "-", "策略手法": strat,
                "倉位狀態": "🛑 數據不足", "期望值 Expectancy": "0.0%", "七維戰術矩陣": "0/7 (無)", "布林通道位階": "數據不足",
                "綜合評級": "D級", "大盤 Alpha (RS20)": "0.0%", "年化 CAGR": "0.0%", 
                "最大回撤 MDD": "0.0%", "夏普比率 Sharpe": "0.00", "平均浮虧 MAE": "0.0%", 
                "平均浮盈 MFE": "0.0%", "建議進場價": "-", "未實現損益": "-", 
                "ATR動態防守價": "-", "複利總報酬": "0.0%", 
                "歷史勝率": "0.0%", "交易次數": 0, "獲利因子": "0.00", "7D得分": 0, "昨日買訊": False
            })
        return stock_reports, {}, err_detail

# --- 7. Session State 記憶庫 ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.final_df = None
    st.session_state.detail_db = {}
    st.session_state.debug_logs = []

# --- 8. 頂部總經抬頭控制卡 ---
col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric("VIX 恐慌指數", f"{vix_score:.2f}", delta="避險戒備" if vix_score >= 25 else "市場平穩", delta_color="inverse")
col_v2.metric("S&P 500 大盤位階", "年線之上 (多頭)" if is_spy_bull else "跌破年線 (空頭)")
col_v3.metric("系統動態總經姿態", market_posture)
st.divider()

if st.button("🚀 啟動 V07.1 美股全自動多因子掃描引擎", use_container_width=True):
    st.session_state.debug_logs = []
    logs = st.session_state.debug_logs
    logs.append(f"🟢 [1] 解析股票代號清單 (共 {len(ticker_list)} 檔): {ticker_list[:5]}...")
    logs.append(f"🟢 [2] 總經環境擷取狀態: {macro_status} | 歷史總經筆數: {len(df_macro)}")

    chunk_size = 25
    ticker_chunks = [ticker_list[i:i + chunk_size] for i in range(0, len(ticker_list), chunk_size)]
    
    master_report = []
    strategies = ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿強勢型", "D: 均值回歸抄底型", "E: 價量動能流跟隨型"]

    for idx_chunk, chunk in enumerate(ticker_chunks):
        logs.append(f"📡 [3.{idx_chunk+1}] 正在向 Yahoo 批量下載第 {idx_chunk+1} 批 (代號: {chunk[:3]}...)")
        try:
            df_chunk = yf.download(chunk, period="2y", progress=False, threads=True)
            logs.append(f"   ➔ 下載完成! df_chunk Shape: {df_chunk.shape} | 是否為空: {df_chunk.empty}")
        except Exception as e:
            df_chunk = pd.DataFrame()
            logs.append(f"   ❌ 第 {idx_chunk+1} 批下載爆發 Exception: {type(e).__name__}: {str(e)}")

        for ticker in chunk:
            df_single = extract_stock_from_chunk(df_chunk, ticker)
            s_reports, s_details, err_status = process_single_stock_us(
                ticker, df_single, cloud_names_dict, backtest_days, df_macro, strategies
            )
            if err_status != "SUCCESS":
                logs.append(f"   ⚠️ 個股 [{ticker}] 異常診斷: {err_status}")

            if s_reports:
                master_report.extend(s_reports)
                st.session_state.detail_db.update(s_details)

    st.session_state.final_df = pd.DataFrame(master_report)
    st.session_state.calculated = True
    st.session_state["scan_time_us"] = datetime.now().strftime("%H:%M:%S")

if st.session_state.get("calculated"):
    st.caption(f"✅ 上次掃描成功時間：{st.session_state.get('scan_time_us', '')}（共呈現 {len(st.session_state.final_df)} 筆機構評估報告）")

if show_debug_log and st.session_state.get("debug_logs"):
    with st.sidebar.expander("🐛 系統診斷日誌", expanded=True):
        st.code("\n".join(st.session_state.debug_logs), language="text")

# --- 9. 🚀 四大專屬主分頁系統 ---
tab_v06_main, tab_7d_bb, tab_verify, tab_debug = st.tabs([
    "📊 倉位動作與狀態分類", 
    "🛡️ 七維矩陣與布林專家診斷", 
    "⚡ 昨日訊號 vs 今日成效驗證",
    "📈 布林通道與 K 線軌跡"
])

with tab_v06_main:
    if st.session_state.calculated and not st.session_state.final_df.empty and "倉位狀態" in st.session_state.final_df.columns:
        st.markdown("### 🎯 **倉位狀態分類面板 (復刻 V06 精準選股觀看)**")
        
        status_tabs = st.tabs([
            "🟢 新進場 / 買入訊號 (BUY)", 
            "📦 獲利續抱中 (HOLD)", 
            "🔴 觸發防守賣出 (SELL)", 
            "💵 空手觀望 / 風控阻擋"
        ])

        df_all = st.session_state.final_df

        # 分類 1: 🟢 新進場
        with status_tabs[0]:
            df_buy = df_all[df_all['倉位狀態'].str.contains("BUY|買入|新進場", na=False)].copy()
            col_b1, col_b2 = st.columns(2)
            col_b1.metric("🟢 當前新進場標的總數", f"{len(df_buy)} 筆")
            try:
                avg_e = df_buy['期望值 Expectancy'].str.rstrip('%').astype(float).mean()
                col_b2.metric("平均期望值", f"{avg_e:+.2f}%")
            except Exception: col_b2.metric("平均期望值", "-")
            st.dataframe(df_buy, use_container_width=True, hide_index=True)

        # 分類 2: 📦 獲利續抱
        with status_tabs[1]:
            df_hold = df_all[df_all['倉位狀態'].str.contains("HOLD|續抱", na=False)].copy()
            col_h1, col_h2 = st.columns(2)
            col_h1.metric("📦 當前獲利續抱標的總數", f"{len(df_hold)} 筆")
            try:
                avg_e = df_hold['期望值 Expectancy'].str.rstrip('%').astype(float).mean()
                col_h2.metric("平均期望值", f"{avg_e:+.2f}%")
            except Exception: col_h2.metric("平均期望值", "-")
            st.dataframe(df_hold, use_container_width=True, hide_index=True)

        # 分類 3: 🔴 觸發防守賣出
        with status_tabs[2]:
            df_sell = df_all[df_all['倉位狀態'].str.contains("SELL|賣出|防守", na=False)].copy()
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("🔴 當前防守離場標的總數", f"{len(df_sell)} 筆")
            try:
                avg_e = df_sell['期望值 Expectancy'].str.rstrip('%').astype(float).mean()
                col_s2.metric("平均期望值", f"{avg_e:+.2f}%")
            except Exception: col_s2.metric("平均期望值", "-")
            st.dataframe(df_sell, use_container_width=True, hide_index=True)

        # 分類 4: 💵 空手觀望
        with status_tabs[3]:
            df_cash = df_all[df_all['倉位狀態'].str.contains("CASH|觀望|赤字|風控", na=False)].copy()
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("💵 當前觀望標的總數", f"{len(df_cash)} 筆")
            try:
                avg_e = df_cash['期望值 Expectancy'].str.rstrip('%').astype(float).mean()
                col_c2.metric("平均期望值", f"{avg_e:+.2f}%")
            except Exception: col_c2.metric("平均期望值", "-")
            st.dataframe(df_cash, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### 📋 **全標的綜合總表 (Master Table)**")
        st.dataframe(st.session_state.final_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 請按下上方「🚀 啟動 V07.1 美股全自動多因子掃描引擎」按鈕開始運算。")

with tab_7d_bb:
    if st.session_state.calculated and not st.session_state.final_df.empty:
        df_all = st.session_state.final_df.copy()

        # 七維戰術矩陣
        st.markdown("## 🛡️ **一、 七維量化戰術矩陣 (全標的評分與分類一覽)**")
        
        with st.expander("📖 **點此展開「七維量化戰術矩陣」代表涵義說明書**", expanded=False):
            st.markdown("""
            * **1. 大盤位階 (200MA)**：評估整體大環境多空姿態，指數在 200MA 年線之上才開啟主要作多倉位。
            * **2. VIX 恐慌位階**：VIX < 22 代表避險情緒平溫，市場流動性充足，策略勝率最高。
            * **3. RSI 區間動能**：14日 RSI 介於 45~75 為多頭健康主升段；<35 為超賣抄底區；>75 注意過熱。
            * **4. 攻擊量能發動**：當日成交量大於 20 日均量，確認有主力實質資金進場推升。
            * **5. MACD 柱狀動能**：MACD 柱狀體翻紅或綠柱連續收斂，代表短期轉折與波段發動。
            * **6. 自由現金流 FCF**：近四季 FCF >= 0 確保企業營運不踩雷，濾除現金流赤字股票。
            * **7. 相對強弱 RS20**：近 20 日個股漲幅超越大盤（Alpha > 0），確保挑選出市場最強領頭羊。
            """)

        m_tabs = st.tabs([
            "🔥 高分極強區 (5~7分)", 
            "⚖️ 常態整理區 (3~4分)", 
            "⚠️ 偏弱觀望區 (0~2分)"
        ])

        with m_tabs[0]:
            df_m_high = df_all[df_all['7D得分'] >= 5]
            st.metric("🔥 高分極強標的數", f"{len(df_m_high)} 筆")
            st.dataframe(df_m_high[['股票代號', '當前市價', '策略手法', '七維戰術矩陣', '倉位狀態', '期望值 Expectancy', '綜合評級']], use_container_width=True, hide_index=True)

        with m_tabs[1]:
            df_m_mid = df_all[(df_all['7D得分'] >= 3) & (df_all['7D得分'] < 5)]
            st.metric("⚖️ 常態整理標的數", f"{len(df_m_mid)} 筆")
            st.dataframe(df_m_mid[['股票代號', '當前市價', '策略手法', '七維戰術矩陣', '倉位狀態', '期望值 Expectancy', '綜合評級']], use_container_width=True, hide_index=True)

        with m_tabs[2]:
            df_m_low = df_all[df_all['7D得分'] < 3]
            st.metric("⚠️ 偏弱觀望標的數", f"{len(df_m_low)} 筆")
            st.dataframe(df_m_low[['股票代號', '當前市價', '策略手法', '七維戰術矩陣', '倉位狀態', '期望值 Expectancy', '綜合評級']], use_container_width=True, hide_index=True)

        st.markdown("#### 🔍 **單股七維戰術明細快速對照**")
        col_tk_7d, col_st_7d = st.columns(2)
        with col_tk_7d: debug_ticker_7d = st.selectbox("🎯 選擇檢視美股代號", ticker_list, key="us_7d_tk")
        with col_st_7d: debug_strat_7d = st.selectbox("🔮 選擇戰術手法", ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿強勢型", "D: 均值回歸抄底型", "E: 價量動能流跟隨型"], key="us_7d_st")
        
        db_key_7d = (debug_ticker_7d, debug_strat_7d)
        if db_key_7d in st.session_state.detail_db:
            m_data = st.session_state.detail_db[db_key_7d]
            m_details = m_data.get("matrix_7d_details", [])
            m_str = m_data.get("matrix_7d_str", "0/7")
            st.info(f"美股 {debug_ticker_7d} ({debug_strat_7d}) - 七維戰術得分：{m_str}")
            if m_details: st.dataframe(pd.DataFrame(m_details), use_container_width=True, hide_index=True)

        st.divider()

        # 布林通道 6 大專家分類 (包含新增的「跌破 20MA 中軌」)
        st.markdown("## 📈 **二、 布林通道 (Bollinger Bands 20,2) 專家診斷與 6 大分類**")
        
        with st.expander("📖 **點此展開「布林通道 (Bollinger Bands)」買賣指令說明書**", expanded=False):
            st.markdown("""
            * **🔥 帶狀極致壓縮 (BB Squeeze)**：通道寬度縮至近 100 日 25% 低位，代表多空能量沉澱至極致，即將爆發大行情。**指令：密切盯盤，準備順勢追擊**。
            * **🚀 突破布林上軌 (Upper Breakout)**：收盤價強勢突破 20MA+2STD 上軌，主力帶量強攻。**指令：動能型買入進場，沿上軌持股續抱**。
            * **🛡️ 貼近 20MA 中軌 (Mid Support)**：多頭趨勢中股價拉回測試 20MA 中軌未跌破。**指令：波段拉回優質加碼點**。
            * **💎 觸及布林下軌 (Lower Bounce)**：股價重摔觸及 20MA-2STD 下軌，超賣落底。**指令：配合紅 K 止跌，均值回歸抄底**。
            * **⚠️ 跌破 20MA 中軌 (離場防守)**：收盤價跌破 20MA 中軌生命線，多頭結構受損。**指令：觸發防守避險，停損或獲利出場**。
            """)

        bb_tabs = st.tabs([
            "🔥 帶狀極致壓縮", 
            "🚀 突破布林上軌", 
            "🛡️ 貼近 20MA 中軌", 
            "💎 觸及布林下軌", 
            "⚠️ 跌破 20MA 中軌",
            "⚖️ 常態通道整理"
        ])

        bb_categories = [
            ("🔥 帶狀極致壓縮", "🔥 帶狀極致壓縮 (準備發動)"),
            ("🚀 突破布林上軌", "🚀 突破布林上軌 (強勢多頭)"),
            ("🛡️ 貼近 20MA 中軌", "🛡️ 貼近 20MA 中軌 (回檔支撐)"),
            ("💎 觸及布林下軌", "💎 觸及布林下軌 (超賣回歸)"),
            ("⚠️ 跌破 20MA 中軌", "⚠️ 跌破 20MA 中軌 (離場防守)"),
            ("⚖️ 常態通道整理", "⚖️ 常態通道內整理")
        ]

        for idx, (tab_title, bb_str_match) in enumerate(bb_categories):
            with bb_tabs[idx]:
                df_bb_sub = df_all[df_all['布林通道位階'] == bb_str_match].copy()
                st.metric(f"{tab_title} 標的數", f"{len(df_bb_sub)} 筆")
                st.dataframe(df_bb_sub[['股票代號', '當前市價', '策略手法', '布林通道位階', '倉位狀態', '期望值 Expectancy', '七維戰術矩陣']], use_container_width=True, hide_index=True)

    else:
        st.info("💡 請先啟動掃描引擎。")

# 新增「昨日訊號 vs 今日成效驗證」分頁
with tab_verify:
    if st.session_state.calculated and not st.session_state.final_df.empty:
        st.markdown("## ⚡ **昨日買訊 vs 今日實質成效驗證 (無偏誤檢驗)**")
        st.caption("驗證昨日 (T-1) 觸發買訊的股票，在今日 (T) 開盤實質成交後的當日盤中與收盤真實表現。")

        df_all = st.session_state.final_df.copy()
        df_yest_buy = df_all[df_all['昨日買訊'] == True].copy()

        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("昨日觸發買訊總數", f"{len(df_yest_buy)} 筆")

        if not df_yest_buy.empty:
            wins = len(df_yest_buy[df_yest_buy['當日驗證'] == "🟢 獲利"])
            win_rate_yest = (wins / len(df_yest_buy)) * 100
            col_v2.metric("今日開盤成交勝率", f"{win_rate_yest:.1f}%")

            try:
                avg_today_ret = df_yest_buy['今日實質漲跌'].str.rstrip('%').astype(float).mean()
                col_v3.metric("今日平均實質報酬", f"{avg_today_ret:+.2f}%")
            except Exception: col_v3.metric("今日平均實質報酬", "-")

            st.dataframe(df_yest_buy[['股票代號', '策略手法', '今日開盤', '今日收盤', '今日實質漲跌', '當日驗證', '期望值 Expectancy', '七維戰術矩陣']], use_container_width=True, hide_index=True)
        else:
            col_v2.metric("今日開盤成交勝率", "-")
            col_v3.metric("今日平均實質報酬", "-")
            st.info("💡 昨日 (T-1 日) 未有任何標的觸發新買訊，市場維持常態觀望。")
    else:
        st.info("💡 請先啟動掃描引擎。")

with tab_debug:
    if st.session_state.calculated:
        col_tk, col_st = st.columns(2)
        with col_tk: debug_ticker = st.selectbox("🎯 選擇美股代號", ticker_list, key="us_debug_tk")
        with col_st: debug_strat = st.selectbox("🔮 選擇策略", ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿強勢型", "D: 均值回歸抄底型", "E: 價量動能流跟隨型"], key="us_debug_st")
        db_key = (debug_ticker, debug_strat)
        if db_key in st.session_state.detail_db:
            data_pack = st.session_state.detail_db[db_key]
            logs_df, buys, sells, v_df = data_pack["logs"], data_pack["buys"], data_pack["sells"], data_pack["v_df"]
            fig = go.Figure()
            if 'BB_Upper' in v_df.columns:
                fig.add_trace(go.Scatter(x=v_df.index, y=v_df['BB_Upper'], mode='lines', name='布林上軌', line=dict(color='rgba(255,165,0,0.6)', width=1, dash='dash')))
                fig.add_trace(go.Scatter(x=v_df.index, y=v_df['BB_Lower'], mode='lines', name='布林下軌', fill='tonexty', fillcolor='rgba(255,165,0,0.06)', line=dict(color='rgba(255,165,0,0.6)', width=1, dash='dash')))
                fig.add_trace(go.Scatter(x=v_df.index, y=v_df['BB_Mid'], mode='lines', name='布林中軌 (20MA)', line=dict(color='rgba(0,191,255,0.7)', width=1.2)))
            
            fig.add_trace(go.Scatter(x=v_df.index, y=v_df['Close'], mode='lines', name='收盤價', line=dict(color='white', width=1.5)))
            
            if len(buys) > 0: fig.add_trace(go.Scatter(x=[b[0] for b in buys], y=[b[1] for b in buys], mode='markers', name='🟢 BUY (T+1成交)', marker=dict(symbol='triangle-up', size=12, color='#00FF00')))
            if len(sells) > 0: fig.add_trace(go.Scatter(x=[s[0] for s in sells], y=[s[1] for s in sells], mode='markers', name='🔴 SELL (ATR防守離場)', marker=dict(symbol='triangle-down', size=12, color='#FF0000')))
            fig.update_layout(title=f"<b>美股 {debug_ticker} - {debug_strat} V07.1 軌跡圖 (含布林通道 20,2)</b>", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            if not logs_df.empty: st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else: st.info("💡 請先啟動掃描引擎。")
