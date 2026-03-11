import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 核心算法部分 ---

def power_law(x, a, b):
    """定义幂函数模型: y = a * x^(-b)"""
    return a * np.power(x, -b)

def fit_retention_curve(days, rates):
    """使用非线性最小二乘法进行拟合"""
    if len(days) < 2:
        return None, None, 0.0
    
    days_arr = np.array(days)
    rates_arr = np.array(rates)
    
    try:
        # 使用 scipy 进行非线性拟合，限制参数范围
        popt, _ = curve_fit(power_law, days_arr, rates_arr, p0=[0.5, 0.5], bounds=(0, [1.0, np.inf]))
        a, b = popt
        
        # 计算拟合优度 R²
        fitted_rates = power_law(days_arr, a, b)
        ss_res = np.sum((rates_arr - fitted_rates) ** 2)
        ss_tot = np.sum((rates_arr - np.mean(rates_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return a, b, r_squared
    except Exception:
        return None, None, 0.0

def get_retention_rate(day, a, b):
    """计算特定天数的留存率"""
    if a is None or b is None: return 0.0
    if day == 0: return 1.0 
    return a * (day ** (-b)) if day > 0 else 0.0

def forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days):
    """DAU 预测核心逻辑"""
    dau_forecast = [current_dau]
    old_dau = current_dau
    
    for t in range(1, forecast_days + 1):
        old_dau *= (1 - churn_rate)
        new_user_contribution = 0
        for prev_t in range(t):
            retention_day = t - (prev_t + 1)
            new_user_contribution += dnu_list[prev_t] * get_retention_rate(retention_day, a, b)
        dau_forecast.append(old_dau + new_user_contribution)
    return dau_forecast

# --- UI 界面部分 ---

st.set_page_config(page_title="App活跃预测工具", layout="wide")
st.title("📱 App用户活跃及 LT 价值预测模型")

# 初始化计算状态
if 'calculated' not in st.session_state:
    st.session_state.calculated = False

col1, col2 = st.columns([1, 3])

with col1:
    st.header("📊 参数输入")
    with st.expander("基础配置", expanded=True):
        current_dau = st.number_input("当前 DAU", value=10000)
        forecast_days = st.slider("预测天数", 7, 90, 30)
        churn_rate = st.number_input("老用户每日流失率 (%)", value=1.0) / 100.0
        daily_dnu = st.number_input("每日平均新增 (DNU)", value=500)
    
    st.subheader("🔄 留存观测点")
    if 'rows' not in st.session_state:
        st.session_state.rows = [{'day': 1, 'rate': 50.0}, {'day': 7, 'rate': 30.0}]

    def add_row():
        st.session_state.rows.append({'day': 30, 'rate': 15.0})

    for i, row in enumerate(st.session_state.rows):
        c1, c2 = st.columns([1, 1])
        st.session_state.rows[i]['day'] = c1.number_input(f"天数", value=row['day'], key=f"d_{i}")
        st.session_state.rows[i]['rate'] = c2.number_input(f"留存%", value=row['rate'], key=f"r_{i}")
    
    st.button("➕ 添加留存点", on_click=add_row)
    
    st.markdown("---")
    # 核心控制：只有点击此按钮，session_state.calculated 才会变为 True
    if st.button("🚀 开始执行预测", type="primary", use_container_width=True):
        st.session_state.calculated = True

with col2:
    st.header("📈 预测分析报告")
    
    if st.session_state.calculated:
        # 提取输入数据
        days_data = [r['day'] for r in st.session_state.rows]
        rates_data = [r['rate'] / 100.0 for r in st.session_state.rows]
        
        # 1. 执行拟合计算
        a, b, r_sq = fit_retention_curve(days_data, rates_data)
        
        if a is not None:
            # 2. 执行 DAU 预测
            dnu_list = [daily_dnu] * (forecast_days + 1)
            forecast_results = forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days)
            
            # 3. 核心指标卡片
            m1, m2, m3 = st.columns(3)
            m1.metric("拟合系数 a", f"{a:.4f}")
            m2.metric("衰减系数 b", f"{b:.4f}")
            m3.metric("拟合优度 R²", f"{r_sq:.4f}")

            # 4. 绘图 (Plotly 整数格式化)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(forecast_results))), 
                y=forecast_results,
                mode='lines+markers',
                name='预测DAU',
                line=dict(color='#3498db', width=3),
                fill='tozeroy',
                hovertemplate="预测天数: %{x}<br>活跃用户数: %{y:,.0f}<extra></extra>"
            ))
            fig.update_layout(
                title=f"未来 {forecast_days} 天 DAU 预测走势",
                xaxis_title="预测天数",
                yaxis_title="活跃用户数 (DAU)",
                hovermode="x unified",
                yaxis=dict(tickformat=",d") 
            )
            st.plotly_chart(fig, use_container_width=True)

            # 5. 长期价值计算 (也被包含在 calculated 判断内)
            st.subheader("🧮 长期价值计算 (LT)")
            lt_n = st.number_input("计算前 N 天的累计留存价值 (LT)", value=30, key="lt_calc_input")
            lt_val = sum(get_retention_rate(d, a, b) for d in range(lt_n + 1))
            st.success(f"根据当前拟合模型，前 {lt_n} 天的累计留存价值 (LT) 为: {lt_val:.2f} 天")
            
        else:
            st.error("拟合失败，请检查观测点输入是否合理。")
    else:
        # 未点击按钮时的引导状态
        st.info("👈 请在左侧配置好当前 DAU、DNU 及历史留存观测点后，点击『开始执行预测』。")

# 依赖文件
with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nscipy\nplotly")
