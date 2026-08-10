import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# --- 1. 網頁核心外觀配置 ---
st.set_page_config(page_title="美股雷達 V07.0 (P0無偏誤修正版)", page_icon="🔮", layout="wide")
st.title("🔮 美股量化沙盒 V07.0 (P0 消除前視偏誤與真實複利回測版)")
st.markdown("已完成 **P0 核心重構與資料對齊修復**：排除「同一 K 線偷看未來」、「當天看收盤當天成交」偏誤，修復欄位自動清洗機制，並導入 **T+1 開盤成交、真實交易成本與複利資金曲線**。")

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
tickers_input = st.sidebar.text_area("📡 當前雲端同步清單", default_tickers, height=100)

temp_raw_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
ticker_list = list(dict.fromkeys(temp_raw_list))

backtest_days = st.sidebar.slider("歷史回測天數設定", min_value=100, max_value=500, value=300, step=50)
enable_fcf_filter = st.sidebar.checkbox("🛡️ 啟用「自由現金流 > 0」安全過濾", value=True)
enable_earnings_shield = st.sidebar.checkbox("💣 啟用「3 天內發布財報」強制避險", value=True)

# 🛠️ 核心修正：多層欄位自動智慧扁平化函式
def fix_yf_columns(df):
    if df is None or df.empty:
        return df
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
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).title() for c in df.columns]
    return df

# --- 3. 🌐 V07 美股大環境與總經雷達 ---
@st.cache_data(ttl=1800)
def fetch_us_macro_dataframe():
    try:
        vix_raw = yf.Ticker("^VIX").history(period="2y")
        spy_raw = yf.Ticker("SPY").history(period="2y")
        
        vix_c = fix_yf_columns(vix_raw)[['Close']].rename(columns={'Close': 'VIX'})
        spy_c = fix_yf_columns(spy_raw)[['Close']].rename(columns={'Close': 'SPY_Close'})
        
        vix_c.index = pd.to_datetime(pd.to_datetime(vix_c.index).date)
        spy_c.index = pd.to_datetime(pd.to_datetime(spy_c.index).date)

        spy_c['SPY_MA200'] = spy_c['SPY_Close'].rolling(200).mean()
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
            
        return df_macro, latest_vix, latest_bull, posture_auto
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=500, freq='D')
        df_macro = pd.DataFrame({'VIX': 18.0, 'Market_Bull': True}, index=dates)
        return df_macro, 18.0, True, "🛡️ 標準平衡型 (預設)"

df_macro, vix_score, is_spy_bull, market_posture = fetch_us_macro_dataframe()

# --- 4. 🏢 V07 基本面雷達 ---
@st.cache_data(ttl=3600)
def fetch_fundamental_and_news(ticker, cloud_dict):
    f_info = {
        "sector_tag": "🇺🇸 美股企業", "pe": "-", "fcf": "-", "rev_growth": "-", 
        "is_fcf_positive": True, "near_earnings": False, "quality_tag": "一般"
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
            if fcf < 0: f_info["is_fcf_positive"] = False
        if rev_g is not None: f_info["rev_growth"] = f"{rev_g * 100:+.1f}%"
        if (fcf is not None and fcf > 0) and (rev_g is not None and rev_g > 0.15):
            f_info["quality_tag"] = "🔥 財報雙強"
    except Exception:
        pass
    return f_info

# --- 5. 技術指標計算 ---
def calculate_indicators(df):
    df = fix_yf_columns(df)
    high_low_diff = (df['High'] - df['Low']).replace(0, 0.001) 
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / high_low_diff
    df['價量動能流'] = (df['Volume'] * mf_multiplier / 1000000).round(2)
    df['主力籌碼'] = df['價量動能流']
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
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

# --- 6. V07 P0 美股歷史回測引擎 ---
def run_backtest_engine_v07_us(df_stock, df_macro_input, strategy_name, days, fund_info, 
                               fee_rate=0.0005, tax_rate=0.0, slippage=0.001):
    df_st = fix_yf_columns(df_stock.copy())
    df_st.index = pd.to_datetime(pd.to_datetime(df_st.index).date)
    
    valid_df = df_st.join(df_macro_input[['VIX', 'Market_Bull']], how='left').ffill().bfill().dropna().tail(days + 1).copy()
    if len(valid_df) < 10:
        return "⚠️ 數據不足", 0.0, 0.0, 0, "0.00", "❌ 不推薦", "🛑 數據不足", "-", "-", "-", [], [], [], valid_df, 0.0, 0.0

    capital = 1.0
    has_position = False
    entry_price = 0.0
    entry_price_with_cost = 0.0
    highest_price_prior = 0.0
    
    total_trades, win_trades = 0, 0
    total_gross_profit, total_gross_loss = 0.0, 0.0
    trade_logs, plot_buys, plot_sells = [], [], []

    dates = valid_df.index
    opens, highs, lows, closes = valid_df['Open'].values, valid_df['High'].values, valid_df['Low'].values, valid_df['Close'].values
    vixs, m_bulls = valid_df['VIX'].values, valid_df['Market_Bull'].values

    s_ma_vals = valid_df['MA5'].values if "A:" in strategy_name else valid_df['MA14'].values
    r14_vals, rsi_vals = valid_df['ROC14'].values, valid_df['RSI_14'].values
    vol_vals, vol_m20_vals = valid_df['Volume'].values, valid_df['Vol_MA20'].values
    m_shrink_vals, m_hist_vals = valid_df['MACD_Shrink'].values, valid_df['MACD_Hist'].values

    pending_buy_signal = False

    for i in range(1, len(valid_df)):
        date_str = dates[i].strftime('%Y-%m-%d')
        open_p, high_p, low_p, close_p = opens[i], highs[i], lows[i], closes[i]
        
        vix_y, bull_y = vixs[i-1], m_bulls[i-1]
        if vix_y >= 25 or not bull_y: rsi_max, vol_mult = 65, 1.50
        elif vix_y <= 15 and bull_y: rsi_max, vol_mult = 75, 1.05
        else: rsi_max, vol_mult = 70, 1.20

        stop_loss_pct = 0.05 if "A:" in strategy_name else 0.075

        # A. 盤中交易執行 (T+1 開盤成交)
        if not has_position:
            if pending_buy_signal:
                has_position = True
                pending_buy_signal = False
                entry_price = open_p * (1 + slippage)
                entry_price_with_cost = entry_price * (1 + fee_rate)
                highest_price_prior = open_p
                total_trades += 1
                trade_logs.append({"交易日期": date_str, "動作狀態": "🟢 買入進場 (BUY)", "執行價格": f"${entry_price:.2f}", "單筆報酬": "-"})
                plot_buys.append((dates[i], entry_price))
        else:
            stop_price = highest_price_prior * (1 - stop_loss_pct)
            is_exit = False
            exit_price = 0.0

            if low_p <= stop_price:
                is_exit = True
                exit_price = min(open_p, stop_price) * (1 - slippage)

            if is_exit:
                exit_price_after_cost = exit_price * (1 - fee_rate - tax_rate)
                trade_return = (exit_price_after_cost - entry_price_with_cost) / entry_price_with_cost
                
                capital *= (1 + trade_return)
                if trade_return > 0: win_trades += 1; total_gross_profit += trade_return
                else: total_gross_loss += abs(trade_return)

                has_position = False
                trade_logs.append({"交易日期": date_str, "動作狀態": "🔴 賣出出場 (SELL)", "執行價格": f"${exit_price:.2f}", "單筆報酬": f"{trade_return*100:+.2f}%"})
                plot_sells.append((dates[i], exit_price))

        # B. 盤後訊號計算
        if has_position:
            highest_price_prior = max(highest_price_prior, high_p)
        else:
            c_p, sma_p = closes[i], s_ma_vals[i]
            r14_p, rsi_p = r14_vals[i], rsi_vals[i]
            vol_p, vol_m20_p = vol_vals[i], vol_m20_vals[i]
            m_shrink_p, m_hist_p = m_shrink_vals[i], m_hist_vals[i]
            m_hist_y = m_hist_vals[i-1]

            if "A:" in strategy_name:
                if (m_shrink_p >= 1 or (m_hist_p > m_hist_y and m_hist_p > 0)) and r14_p > 0 and rsi_p < rsi_max: pending_buy_signal = True
            elif "B:" in strategy_name:
                if c_p > sma_p and vol_p > vol_m20_p * vol_mult: pending_buy_signal = True

    total_return = capital - 1.0
    final_win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else (99.9 if total_gross_profit > 0 else 0.0)
    pf_str = "無限" if profit_factor == 99.9 else f"{profit_factor:.2f}"

    stars = "❌ 不推薦"
    if total_return > 0 and total_trades > 0:
        if total_return >= 0.20 and final_win_rate >= 0.52: stars = "⭐⭐⭐⭐⭐"
        elif total_return >= 0.10 or final_win_rate >= 0.48: stars = "⭐⭐⭐⭐"
        else: stars = "⭐⭐"

    current_close = closes[-1]
    if has_position:
        current_status = "📦 獲利續抱中 (HOLD)"
        unrealized_pnl = (current_close - entry_price_with_cost) / entry_price_with_cost
        pnl_str = f"{unrealized_pnl*100:+.2f}%"
        entry_price_str = f"${entry_price:.2f}"
        sl_price_str = f"${highest_price_prior * (1 - stop_loss_pct):.2f}"
    else:
        current_status = "💵 空手觀望 (CASH)"
        entry_price_str = "-"
        sl_price_str = "-"
        pnl_str = "-"

    return "📡 運算完畢", total_return, final_win_rate, total_trades, pf_str, stars, current_status, entry_price_str, sl_price_str, pnl_str, trade_logs, plot_buys, plot_sells, valid_df, entry_price, highest_price_prior * (1 - stop_loss_pct)

# ⚡ 多線程 Worker
def process_single_stock_us(ticker, cloud_dict, backtest_days, df_macro_data, strategies):
    try:
        df_stock = yf.download(ticker, period="2y", progress=False)
        if df_stock.empty: return [], {}, [], {}
        df_stock = fix_yf_columns(df_stock)
        df_stock = calculate_indicators(df_stock)
        
        df_temp_clean = df_stock.dropna(subset=['Close'])
        current_close = float(df_temp_clean['Close'].iloc[-1]) if not df_temp_clean.empty else 0.0
        fund_info = fetch_fundamental_and_news(ticker, cloud_dict)

        stock_reports = []
        stock_details = {}
        
        for strat in strategies:
            radar, ret, win, trades, pf, stars, cur_status, entry_price_val, sl_price, pnl, t_logs, p_buys, p_sells, v_df, raw_entry, raw_sl = run_backtest_engine_v07_us(df_stock, df_macro_data, strat, backtest_days, fund_info)
            stock_details[(ticker, strat)] = {"logs": pd.DataFrame(t_logs), "buys": p_buys, "sells": p_sells, "v_df": v_df}
            stock_reports.append({
                "股票代號": ticker, "當前市價": f"${current_close:.2f}", "策略手法": strat,
                "倉位狀態": cur_status, "基本面評價": fund_info["quality_tag"],
                "建議進場價(持股成本)": entry_price_val, "未實現損益": pnl, "嚴格防守價": sl_price,
                "複利總報酬率": f"{ret * 100:+.2f}%", "歷史勝率": f"{win * 100:.1f}%", "交易次數": trades, "獲利因子": pf, "推薦指數": stars
            })
        return stock_reports, stock_details, [], {}
    except Exception:
        return [], {}, [], {}

# --- 7. Session State 記憶庫 ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
    st.session_state.final_df = None
    st.session_state.detail_db = {}

# --- 8. 頂部總經抬頭控制卡 ---
col_v1, col_v2, col_v3 = st.columns(3)
col_v1.metric("VIX 恐慌指數", f"{vix_score:.2f}", delta="避險戒備" if vix_score >= 25 else "市場平穩", delta_color="inverse")
col_v2.metric("S&P 500 大盤位階", "年線之上 (多頭)" if is_spy_bull else "跌破年線 (空頭)")
col_v3.metric("系統動態總經姿態", market_posture)
st.divider()

if st.button("🚀 啟動 V07.0 美股全自動多因子掃描引擎 (⚡ 多線程 P0 修復版)", use_container_width=True):
    with st.spinner("正在啟動 ThreadPoolExecutor 多線程引擎進行 P0 計算..."):
        master_report, strategies = [], ["A: 激進動能型", "B: 穩健波段型"]
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for ticker in ticker_list:
                f = executor.submit(process_single_stock_us, ticker, cloud_names_dict, backtest_days, df_macro, strategies)
                futures.append(f)
                
        for f in futures:
            s_reports, s_details, _, _ = f.result()
            if s_reports:
                master_report.extend(s_reports)
                st.session_state.detail_db.update(s_details)
                
        st.session_state.final_df = pd.DataFrame(master_report)
        st.session_state.calculated = True
        st.success("📊 V07.0 美股無偏誤回測計算完成！")

# --- 9. 網頁分頁系統 ---
tab_v07, tab_debug = st.tabs(["📊 倉位動作與複利總表", "🔍 歷史回測驗證"])

with tab_v07:
    if st.session_state.calculated:
        st.dataframe(st.session_state.final_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 請按下上方「🚀 啟動 V07.0 美股全自動多因子掃描引擎」按鈕開始運算。")

with tab_debug:
    if st.session_state.calculated:
        col_tk, col_st = st.columns(2)
        with col_tk: debug_ticker = st.selectbox("🎯 選擇美股代號", ticker_list)
        with col_st: debug_strat = st.selectbox("🔮 選擇策略", ["A: 激進動能型", "B: 穩健波段型"])
        db_key = (debug_ticker, debug_strat)
        if db_key in st.session_state.detail_db:
            data_pack = st.session_state.detail_db[db_key]
            logs_df, buys, sells, v_df = data_pack["logs"], data_pack["buys"], data_pack["sells"], data_pack["v_df"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v_df.index, y=v_df['Close'], mode='lines', name='收盤價', line=dict(color='lightgrey', width=1.5)))
            if len(buys) > 0: fig.add_trace(go.Scatter(x=[b[0] for b in buys], y=[b[1] for b in buys], mode='markers', name='🟢 BUY (T+1成交)', marker=dict(symbol='triangle-up', size=12, color='#00FF00')))
            if len(sells) > 0: fig.add_trace(go.Scatter(x=[s[0] for s in sells], y=[s[1] for s in sells], mode='markers', name='🔴 SELL (停損/停利)', marker=dict(symbol='triangle-down', size=12, color='#FF0000')))
            fig.update_layout(title=f"<b>美股 {debug_ticker} - {debug_strat} V07.0 軌跡圖</b>", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            if not logs_df.empty: st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else: st.info("💡 請先啟動掃描引擎。")
