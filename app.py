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
st.set_page_config(page_title="美股雷達 V07.3 (P3解包修復版)", page_icon="🔮", layout="wide")
st.title("🔮 美股量化沙盒 V07.3 (P3 機構級統計與全防護版)")
st.markdown("已完成 **P3 評估重構與解包 Bug 修正**：實裝 **Expectancy 期望值**、**CAGR**、**MDD**、**Sharpe** 與 **MAE/MFE 診斷**。")

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
tickers_input = st.sidebar.text_area("📡 當前雲端同步清單", default_tickers, height=120)

temp_raw_list = [t.strip().upper() for t in re.split(r'[\n\r,\s]+', tickers_input) if t.strip()]
ticker_list = list(dict.fromkeys(temp_raw_list))

backtest_days = st.sidebar.slider("歷史回測天數設定", min_value=100, max_value=500, value=300, step=50)
enable_fcf_filter = st.sidebar.checkbox("🛡️ 啟用「FCF 負值」強制攔截", value=True)

# 🛠️ 萬能欄位清洗函式
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

# 🛠️ 單股抽取函式
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

# --- 5. 技術指標計算 ---
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
    
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
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

# --- 6. V07 P3 美股歷史回測引擎 ---
def run_backtest_engine_v07_us(df_stock, df_macro_input, strategy_name, days, fund_info, 
                               fee_rate=0.0005, tax_rate=0.0, slippage=0.001):
    df_st = clean_and_flatten_df(df_stock.copy())
    df_st.index = pd.to_datetime(pd.to_datetime(df_st.index).date)
    
    valid_df = df_st.join(df_macro_input[['VIX', 'Market_Bull', 'SPY_Close']], how='left').ffill().bfill().dropna().tail(days + 1).copy()
    
    if len(valid_df) < 10:
        return ("⚠️ 數據不足", 0.0, 0.0, 0, "0.00", "D級", "🛑 數據不足", "-", "-", "-", 
                [], [], [], valid_df, 0.0, 0.0, "0.0%", "0.0%", "0.0%", "0.0%", "0.00", "0.0%", "0.0%")

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
    pv_flow_vals, q80_vals = valid_df['價量動能流'].values, valid_df['動能流_Q80'].values
    rs_vals = valid_df['RS_20'].values

    pending_buy_signal = False

    for i in range(1, len(valid_df)):
        date_str = dates[i].strftime('%Y-%m-%d')
        open_p, high_p, low_p, close_p = opens[i], highs[i], lows[i], closes[i]
        
        vix_y, bull_y = vixs[i-1], m_bulls[i-1]
        # 🛠️ 核心修復：修正 else 賦值少寫 dip_pct 的 Bug
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
            m50_y = m50_vals[i-3] if i >= 3 else m50_p

            if "A:" in strategy_name:
                if (m_shrink_p >= 1 or (m_hist_p > m_hist_y and m_hist_p > 0)) and r14_p > 0 and rsi_p < rsi_max: pending_buy_signal = True
            elif "B:" in strategy_name:
                if c_p > sma_p and vol_p > vol_m20_p * vol_mult and clv_p >= 0.65 and rs_p > 0: pending_buy_signal = True
            elif "C:" in strategy_name:
                if c_p > sma_p and vol_p > vol_m20_p * (vol_mult * 1.1) and clv_p >= 0.70 and rs_p > 0.02: pending_buy_signal = True
            elif "D:" in strategy_name:
                if m200_p > 0 and (c_p - m200_p)/m200_p <= dip_pct and rsi_p < 35 and m_shrink_p >= 1 and c_p > opens[i]: pending_buy_signal = True
            elif "E:" in strategy_name:
                if pv_flow_p > q80_p and pv_flow_p > 0 and c_p > m50_p and m50_p >= m50_y and rs_p > 0: pending_buy_signal = True

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
    else:
        current_status = "💵 空手觀望 (CASH)"
        entry_price_str = "-"
        sl_price_str = "-"
        pnl_str = "-"

    if enable_fcf_filter and fund_info["fcf_status"] == "NEGATIVE" and "HOLD" in current_status:
        current_status = "⚠️ 現金流赤字/風控阻擋 (CASH)"

    latest_rs = rs_vals[-1] * 100 if len(rs_vals) > 0 else 0.0
    rs_tag = f"{latest_rs:+.1f}%"

    return ("📡 運算完畢", total_return, win_rate, total_trades, pf_str, grade, 
            current_status, entry_price_str, sl_price_str, pnl_str, trade_logs, 
            plot_buys, plot_sells, valid_df, entry_price, current_stop_price, rs_tag,
            f"{expectancy*100:+.2f}%", f"{cagr*100:+.1f}%", f"{mdd*100:.1f}%", 
            f"{sharpe:.2f}", f"{avg_mae*100:.1f}%", f"{avg_mfe*100:+.1f}%")

# ⚡ 單個股票處理 Worker
def process_single_stock_us(ticker, df_stock, cloud_dict, backtest_days, df_macro_data, strategies):
    try:
        if df_stock is None or df_stock.empty or len(df_stock) < 10:
            stock_reports = []
            for strat in strategies:
                stock_reports.append({
                    "股票代號": ticker, "當前市價": "-", "策略手法": strat,
                    "倉位狀態": "🛑 數據不足", "期望值 Expectancy": "0.0%",
                    "綜合評級": "D級", "大盤 Alpha (RS20)": "0.0%", "年化 CAGR": "0.0%", 
                    "最大回撤 MDD": "0.0%", "夏普比率 Sharpe": "0.00", "平均浮虧 MAE": "0.0%", 
                    "平均浮盈 MFE": "0.0%", "建議進場價": "-", "未實現損益": "-", 
                    "ATR動態防守價": "-", "複利總報酬": "0.0%", 
                    "歷史勝率": "0.0%", "交易次數": 0, "獲利因子": "0.00"
                })
            return stock_reports, {}, "K線數據為空或長度不足 (<10)"

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
             rs_tag, expectancy_str, cagr_str, mdd_str, sharpe_str, mae_str, mfe_str) = run_backtest_engine_v07_us(
                df_stock, df_macro_data, strat, backtest_days, fund_info
            )
            
            stock_details[(ticker, strat)] = {"logs": pd.DataFrame(t_logs), "buys": p_buys, "sells": p_sells, "v_df": v_df}
            fcf_disp = fund_info["fcf"] if fund_info["fcf_status"] != "UNKNOWN" else "❓ 數據缺失"

            stock_reports.append({
                "股票代號": ticker, "當前市價": f"${current_close:.2f}", "策略手法": strat,
                "倉位狀態": cur_status, "期望值 Expectancy": expectancy_str,
                "綜合評級": grade, "大盤 Alpha (RS20)": rs_tag, "年化 CAGR": cagr_str, 
                "最大回撤 MDD": mdd_str, "夏普比率 Sharpe": sharpe_str, "平均浮虧 MAE": mae_str, 
                "平均浮盈 MFE": mfe_str, "建議進場價": entry_price_val, "未實現損益": pnl, 
                "ATR動態防守價": sl_price, "複利總報酬": f"{ret * 100:+.2f}%", 
                "歷史勝率": f"{win * 100:.1f}%", "交易次數": trades, "獲利因子": pf
            })
        return stock_reports, stock_details, "SUCCESS"
    except Exception as e:
        err_msg = f"EXCEPTION: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        stock_reports = []
        for strat in strategies:
            stock_reports.append({
                "股票代號": ticker, "當前市價": "-", "策略手法": strat,
                "倉位狀態": "🛑 數據不足", "期望值 Expectancy": "0.0%",
                "綜合評級": "D級", "大盤 Alpha (RS20)": "0.0%", "年化 CAGR": "0.0%", 
                "最大回撤 MDD": "0.0%", "夏普比率 Sharpe": "0.00", "平均浮虧 MAE": "0.0%", 
                "平均浮盈 MFE": "0.0%", "建議進場價": "-", "未實現損益": "-", 
                "ATR動態防守價": "-", "複利總報酬": "0.0%", 
                "歷史勝率": "0.0%", "交易次數": 0, "獲利因子": "0.00"
            })
        return stock_reports, {}, err_msg

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

if st.button("🚀 啟動 V07.3 美股全自動多因子掃描引擎", use_container_width=True):
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
            logs.append(f"   ❌ 第 {idx_chunk+1} 批下載爆發 Exception: {str(e)}")

        for ticker in chunk:
            df_single = extract_stock_from_chunk(df_chunk, ticker)
            s_reports, s_details, err_code = process_single_stock_us(
                ticker, df_single, cloud_names_dict, backtest_days, df_macro, strategies
            )
            if err_code != "SUCCESS":
                logs.append(f"   ⚠️ 個股 [{ticker}] 運算異常原因: {err_code}")
            
            if s_reports:
                master_report.extend(s_reports)
                st.session_state.detail_db.update(s_details)

    st.session_state.final_df = pd.DataFrame(master_report)
    st.session_state.calculated = True
    st.success(f"🎉 掃描完成！已呈現 {len(st.session_state.final_df)} 筆報告！")

# 🐛 系統診斷 Log 展示區塊
if st.session_state.get("debug_logs"):
    with st.expander("🐛 系統診斷日誌", expanded=False):
        st.code("\n".join(st.session_state.debug_logs), language="text")

# --- 9. 網頁分頁系統 ---
tab_v07, tab_debug = st.tabs(["📊 倉位動作與機構級統計總表", "🔍 歷史回測驗證"])

with tab_v07:
    if st.session_state.calculated:
        st.dataframe(st.session_state.final_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 請按下上方「🚀 啟動 V07.3 美股全自動多因子掃描引擎」按鈕開始運算。")

with tab_debug:
    if st.session_state.calculated:
        col_tk, col_st = st.columns(2)
        with col_tk: debug_ticker = st.selectbox("🎯 選擇美股代號", ticker_list)
        with col_st: debug_strat = st.selectbox("🔮 選擇策略", ["A: 激進動能型", "B: 穩健波段型", "C: 槓桿強勢型", "D: 均值回歸抄底型", "E: 價量動能流跟隨型"])
        db_key = (debug_ticker, debug_strat)
        if db_key in st.session_state.detail_db:
            data_pack = st.session_state.detail_db[db_key]
            logs_df, buys, sells, v_df = data_pack["logs"], data_pack["buys"], data_pack["sells"], data_pack["v_df"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=v_df.index, y=v_df['Close'], mode='lines', name='收盤價', line=dict(color='lightgrey', width=1.5)))
            if len(buys) > 0: fig.add_trace(go.Scatter(x=[b[0] for b in buys], y=[b[1] for b in buys], mode='markers', name='🟢 BUY (T+1成交)', marker=dict(symbol='triangle-up', size=12, color='#00FF00')))
            if len(sells) > 0: fig.add_trace(go.Scatter(x=[s[0] for s in sells], y=[s[1] for s in sells], mode='markers', name='🔴 SELL (ATR防守離場)', marker=dict(symbol='triangle-down', size=12, color='#FF0000')))
            fig.update_layout(title=f"<b>美股 {debug_ticker} - {debug_strat} V07.3 軌跡圖</b>", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            if not logs_df.empty: st.dataframe(logs_df, use_container_width=True, hide_index=True)
    else: st.info("💡 請先啟動掃描引擎。")
