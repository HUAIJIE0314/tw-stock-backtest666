import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 0. 網頁基本設定
# ==========================================
st.set_page_config(page_title="台股 60 分 K 趨勢策略回測", page_icon="📈", layout="wide")

# ==========================================
# 1. 資料抓取模組 (使用快取避免重複發送請求)
# ==========================================
@st.cache_data(ttl=86400) # 快取一天
def get_all_tw_stocks_with_names():
    """使用政府 Open API 抓取最新股票代號與【中文簡稱】的字典"""
    def extract_codes_and_names(api_url, suffix):
        try:
            res = requests.get(api_url, timeout=10)
            data = res.json()
            if not data: return {}
            
            sample_keys = data[0].keys()
            code_key = next((k for k in sample_keys if k in ['公司代號', '證券代號', '股票代號', 'Code', 'code', 'SecuritiesCompanyCode', 'Symbol']), None)
            name_key = next((k for k in sample_keys if k in ['公司簡稱', '證券名稱', '公司名稱', 'Name', 'name', 'CompanyAbbreviation', 'CompanyName']), None)
            
            if not code_key or not name_key: return {}
                
            stock_dict = {}
            for item in data:
                code = str(item.get(code_key, '')).strip()
                name = str(item.get(name_key, '')).strip()
                if code and len(code) == 4: 
                    stock_dict[f"{code}{suffix}"] = name
            return stock_dict
        except Exception:
            return {}

    twse_dict = extract_codes_and_names("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW")
    tpex_dict = extract_codes_and_names("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", ".TWO")
    
    full_stock_dict = {**twse_dict, **tpex_dict}
    return full_stock_dict if full_stock_dict else {'2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科'}

# ==========================================
# 2. 側邊欄：參數設定
# ==========================================
st.sidebar.header("⚙️ 回測參數設定")

user_ticker = st.sidebar.text_input("股票代號 (不需加 .TW)", value="2337", max_chars=4)
initial_capital = st.sidebar.number_input("投入本金 (元)", min_value=10000, max_value=10000000, value=500000, step=10000)
day_interval = st.sidebar.number_input("Day Interval (尋找金叉天數)", min_value=1, max_value=20, value=3, step=1)
ma_sell = st.sidebar.number_input("賣出均線 (MA_sell)", min_value=5, max_value=240, value=60, step=1)
backtest_days = st.sidebar.slider("回測天數 (yfinance 60m 限制 730 天)", min_value=30, max_value=730, value=120, step=10)

if st.sidebar.button("🚀 執行回測", use_container_width=True):
    
    with st.spinner('正在下載資料與計算中...'):
        # 處理股票代號與名稱
        stock_dict = get_all_tw_stocks_with_names()
        filtered_list = {k: v for k, v in stock_dict.items() if k.startswith(user_ticker)}
        
        if not filtered_list:
            st.error(f"找不到符合條件的股票代號：{user_ticker}")
            st.stop()
            
        ticker_full = list(filtered_list.keys())[0]
        stock_name = filtered_list[ticker_full]
        
        # 演算法固定參數
        RSI_PERIOD = 14
        KD_K, KD_D, KD_SMOOTH = 60, 3, 3
        bars_per_day = 5
        lookback_bars = day_interval * bars_per_day
        MA_select = f"{ma_sell}MA"

        # 下載資料
        df = yf.download(ticker_full, period=f"{backtest_days}d", interval="60m", progress=False)
        
        if df.empty:
            st.error("無法取得該股票的歷史資料，可能已下市或 API 暫時錯誤。")
            st.stop()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 時區處理
        try:
            df.index = df.index.tz_convert('Asia/Taipei')
        except TypeError:
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Taipei')

        # 計算技術指標
        df['RSI'] = df.ta.rsi(length=RSI_PERIOD)
        df['5MA'] = df.ta.sma(length=5)
        df['60MA'] = df.ta.sma(length=60) # 為了保留你原本程式的 MA_Cross 判斷
        if MA_select != '60MA':
            df[MA_select] = df.ta.sma(length=ma_sell)
            
        stoch = df.ta.stoch(k=KD_K, d=KD_D, smooth_k=KD_SMOOTH)
        df = pd.concat([df, stoch], axis=1)

        k_col = [col for col in df.columns if 'STOCHk' in col][0]
        d_col = [col for col in df.columns if 'STOCHd' in col][0]

        # 產生進出場訊號 (嚴格複製)
        df['KD_Cross'] = (df[k_col] > 50) & (df[k_col].shift(1) <= 50)
        # 買進條件使用 60MA 作為基準 (與原代碼一致)
        df['MA_Cross'] = (df['5MA'] > df['60MA']) & (df['5MA'].shift(1) <= df['60MA'].shift(1))
        df['KD_Cross_5d'] = df['KD_Cross'].rolling(window=lookback_bars).max()
        df['MA_Cross_5d'] = df['MA_Cross'].rolling(window=lookback_bars).max()

        df['MA_Golden_Cross'] = (df['KD_Cross_5d'] == 1) & (df['MA_Cross_5d'] == 1)
        df['Buy_Signal'] = (df['RSI'] > 60) & (df['MA_Golden_Cross'])

        # 賣出條件使用可變動的 MA_select (與原代碼一致)
        df['MA_Death_Cross'] = (df['5MA'] < df[MA_select]) & (df['5MA'].shift(1) >= df[MA_select].shift(1))
        df['Sell_Signal'] = (df['MA_Death_Cross']) 

        df = df.dropna()

        # 執行回測邏輯 (事件驅動)
        capital = initial_capital  
        position = 0               
        entry_price = 0.0          
        entry_date = None
        equity_curve = [initial_capital] 
        trade_history = []         
        buy_points = []
        sell_points = []

        for date, row in df.iterrows():
            current_price = row['Close']
            
            # 尋找買點
            if position == 0:
                if row['Buy_Signal']:
                    shares_to_buy = int(capital / (current_price * 1.001425))
                    if shares_to_buy > 0:
                        position = shares_to_buy
                        entry_price = current_price
                        entry_date = date
                        capital = capital - (position * entry_price * 1.001425)
                        buy_points.append({'Date': date, 'Price': entry_price})
                
            # 判斷出場
            elif position > 0:
                if row['Sell_Signal']:
                    sell_revenue = position * current_price * (1 - 0.001425 - 0.003)
                    capital += sell_revenue
                    
                    total_cost = position * entry_price * 1.001425
                    profit = sell_revenue - total_cost
                    profit_pct = (profit / total_cost) * 100
                    
                    trade_history.append({
                        'Buy_Date': entry_date,     
                        'Sell_Date': date,          
                        'Buy_Price': entry_price,
                        'Sell_Price': current_price,
                        'Profit': profit,
                        'Return(%)': round(profit_pct, 2)
                    })
                    sell_points.append({'Date': date, 'Price': current_price})
                    
                    position = 0
                    entry_price = 0.0
                    entry_date = None

            current_equity = capital + (position * current_price * (1 - 0.001425 - 0.003))
            equity_curve.append(current_equity)

        equity_curve = equity_curve[1:]
        df['Equity_Curve'] = equity_curve

        # 計算統計數據
        total_return = (df['Equity_Curve'].iloc[-1] - initial_capital) / initial_capital
        total_trades = len(trade_history) 
        winning_trades = sum(1 for t in trade_history if t['Profit'] > 0)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0

        # ==========================================
        # 3. 畫面呈現：回測報告與數據
        # ==========================================
        st.header(f"📊 {user_ticker} {stock_name} - 策略回測報告")
        
        # 頂部指標卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("初始資金", f"{initial_capital:,.0f} 元")
        col2.metric("最終淨值", f"{df['Equity_Curve'].iloc[-1]:,.0f} 元", f"{total_return * 100:.2f}%")
        col3.metric("勝率", f"{win_rate * 100:.2f}%")
        col4.metric("總交易次數", f"{total_trades} 次")

        # 未平倉部位顯示
        if position > 0:
            final_price = df['Close'].iloc[-1]
            current_stock_value = position * final_price * (1 - 0.001425 - 0.003)
            initial_cost = position * entry_price * 1.001425
            unrealized_profit = current_stock_value - initial_cost
            unrealized_pct = (final_price - entry_price) / entry_price * 100
            
            with st.expander("⚠️ 檢視目前持有未平倉部位 (截至回測最後一筆)", expanded=True):
                st.info(f"**買入時間:** {entry_date.strftime('%Y/%m/%d %H:%M')} | **買入價格:** {entry_price:.2f} 元 | **持有股數:** {position:,} 股")
                
                # 利用顏色標示未實現損益
                profit_color = "normal" if unrealized_profit == 0 else "inverse" if unrealized_profit < 0 else "normal"
                st.metric("預估未實現損益", f"{unrealized_profit:,.0f} 元", f"{unrealized_pct:.2f}%", delta_color=profit_color)

        # ==========================================
        # 4. 畫面呈現：Plotly 互動式圖表 (非常適合手機)
        # ==========================================
        st.subheader("🎨 策略視覺化圖表")
        
        # 建立 4 個子圖，共用 X 軸
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.4, 0.15, 0.25, 0.2],
            subplot_titles=("價格與均線", "RSI (14)", "KD (60, 3, 3)", "總資金曲線")
        )


        # # 原本：
        # # fig.add_trace(go.Scatter(x=df.index, y=df['Close'], ...))

        # # 改成：
        # x_strings = df.index.strftime('%m/%d %H:%M')
        # fig.add_trace(go.Scatter(x=x_strings, y=df['Close'], name='Close Price', line=dict(color='#d1d5db', width=2)), row=1, col=1)


        # x_strings = df.index.strftime('%m/%d %H:%M')
        x_strings = df.index.strftime('%Y/%m/%d %H:%M')
        # --- 1. 價格圖 (ax1) ---
        fig.add_trace(go.Scatter(x=x_strings, y=df['Close'], name='Close Price', line=dict(color='#d1d5db', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_strings, y=df['5MA'], name='5MA', line=dict(color='#3b82f6', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_strings, y=df[MA_select], name=MA_select, line=dict(color='#ef4444', width=1)), row=1, col=1)

        # 標記買賣點
        # if buy_points:
        #     b_dates = [b['Date'] for b in buy_points]
        #     b_prices = [b['Price'] for b in buy_points]
        #     fig.add_trace(go.Scatter(x=b_dates, y=b_prices, mode='markers', name='Buy',
        #                              marker=dict(symbol='triangle-up', size=14, color='#22c55e')), row=1, col=1)
        # if sell_points:
        #     s_dates = [s['Date'] for s in sell_points]
        #     s_prices = [s['Price'] for s in sell_points]
        #     fig.add_trace(go.Scatter(x=s_dates, y=s_prices, mode='markers', name='Sell',
        #                              marker=dict(symbol='triangle-down', size=14, color='#f97316')), row=1, col=1)
        # 標記買賣點 (🌟 把 Date 轉成與 X 軸一模一樣的字串)
        if buy_points:
            b_dates = [b['Date'].strftime('%Y/%m/%d %H:%M') for b in buy_points] 
            b_prices = [b['Price'] for b in buy_points]
            fig.add_trace(go.Scatter(x=b_dates, y=b_prices, mode='markers', name='Buy',
                                     marker=dict(symbol='triangle-up', size=14, color='#22c55e')), row=1, col=1)
        if sell_points:
            s_dates = [s['Date'].strftime('%Y/%m/%d %H:%M') for s in sell_points]
            s_prices = [s['Price'] for s in sell_points]
            fig.add_trace(go.Scatter(x=s_dates, y=s_prices, mode='markers', name='Sell',
                                     marker=dict(symbol='triangle-down', size=14, color='#f97316')), row=1, col=1)

        # --- 2. RSI 圖 (ax2) ---
        fig.add_trace(go.Scatter(x=x_strings, y=df['RSI'], name='RSI', line=dict(color='#8b5cf6', width=1.5)), row=2, col=1)
        fig.add_hline(y=60, line_dash="dash", line_color="#22c55e", row=2, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="#f97316", row=2, col=1)

        # --- 3. KD 圖 (ax3) ---
        fig.add_trace(go.Scatter(x=x_strings, y=df[k_col], name='K (60,3)', line=dict(color='#f59e0b', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_strings, y=df[d_col], name='D (3)', line=dict(color='#0ea5e9', width=1.5)), row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", row=3, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="#6b7280", opacity=0.5, row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#22c55e", row=3, col=1)

        # --- 4. 資金曲線圖 (ax4) ---
        fig.add_trace(go.Scatter(x=x_strings, y=df['Equity_Curve'], name='Total Equity', line=dict(color='#10b981', width=2)), row=4, col=1)

        # --- 畫出垂直買賣貫穿線 ---
        # if buy_points:
        #     for b in buy_points:
        #         fig.add_vline(x=b['Date'], line_width=1, line_dash="dash", line_color="#22c55e", opacity=0.5)
        # if sell_points:
        #     for s in sell_points:
        #         fig.add_vline(x=s['Date'], line_width=1, line_dash="dash", line_color="#f97316", opacity=0.5)

        # --- 畫出垂直買賣貫穿線 (🌟 同樣要轉成字串) ---
        if buy_points:
            for b in buy_points:
                date_str = b['Date'].strftime('%Y/%m/%d %H:%M')
                fig.add_vline(x=date_str, line_width=1, line_dash="dash", line_color="#22c55e", opacity=0.5)
        if sell_points:
            for s in sell_points:
                date_str = s['Date'].strftime('%Y/%m/%d %H:%M')
                fig.add_vline(x=date_str, line_width=1, line_dash="dash", line_color="#f97316", opacity=0.5)


        # 圖表版面設定 (過濾掉非交易時間的空白)
        fig.update_layout(
            height=900, 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            # showlegend=False # 手機上圖例太佔空間，直接用游標/點擊確認即可
            showlegend=True,  # 🌟 1. 這裡改成 True 把它叫回來
            legend=dict(      # 🌟 2. 新增這段，讓圖例水平排列在圖表正上方
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        # 過濾六日空白 (如需過濾盤後時間也可加在此)
        # fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        # fig.update_xaxes(
        #     rangebreaks=[
        #         dict(bounds=["sat", "mon"]), # 隱藏週末 (星期六到星期一早)
        #         dict(bounds=[13.5, 9], pattern="hour") # 隱藏盤後 (13:30 到 09:00)
        #     ]
        # )

        fig.update_xaxes(type='category', nticks=15)

        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # 5. 交易明細
        # ==========================================
        st.subheader("📋 交易明細")
        if not trade_history:
            st.write("這段期間內無任何交易。")
        else:
            # 整理 DataFrame 並格式化顯示
            df_trades = pd.DataFrame(trade_history)
            df_trades['Buy_Date'] = df_trades['Buy_Date'].dt.strftime('%Y/%m/%d %H:%M')
            df_trades['Sell_Date'] = df_trades['Sell_Date'].dt.strftime('%Y/%m/%d %H:%M')
            
            # 使用 Streamlit 原生的 dataframe 表格展示 (支援排序、捲動)
            st.dataframe(
                df_trades.style.format({
                    "Buy_Price": "{:.2f}",
                    "Sell_Price": "{:.2f}",
                    "Profit": "{:,.0f}",
                    "Return(%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )