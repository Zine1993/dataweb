import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

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
        popt, _ = curve_fit(power_law, days_arr, rates_arr, p0=[0.5, 0.5], bounds=(0, [1.0, np.inf]))
        a, b = popt
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

# 1. 初始化 Session State
if 'rows' not in st.session_state:
    st.session_state.rows = [{'day': 1, 'rate': 50.0}, {'day': 7, 'rate': 30.0}]
if 'fitted_params' not in st.session_state:
    st.session_state.fitted_params = None  # 用于锁定计算结果

col1, col2 = st.columns([1, 3])

with col1:
    st.header("📊 参数输入")
    with st.expander("基础配置", expanded=True):
        current_dau = st.number_input("当前 DAU", value=10000)
        forecast_days = st.slider("预测天数", 7, 90, 30)
        churn_rate = st.number_input("老用户每日流失率 (%)", value=1.0) / 100.0
        daily_dnu = st.number_input("每日平均新增 (DNU)", value=500)
    
    st.subheader("🔄 留存观测点")
    
    # 动态渲染留存点输入及删除按钮
    for i, row in enumerate(st.session_state.rows):
        c1, c2, c3 = st.columns([1, 1, 0.4])
        st.session_state.rows[i]['day'] = c1.number_input(f"天数", value=row['day'], key=f"d_{i}")
        st.session_state.rows[i]['rate'] = c2.number_input(f"留存%", value=row['rate'], key=f"r_{i}")
        if c3.button("🗑️", key=f"del_{i}"):
            st.session_state.rows.pop(i)
            st.rerun()
    
    if st.button("➕ 添加留存点"):
        st.session_state.rows.append({'day': 30, 'rate': 15.0})
        st.rerun()
    
    st.markdown("---")
    # 核心修复：点击按钮才进行拟合并将结果锁存到 session_state
    if st.button("🚀 开始执行预测", type="primary", use_container_width=True):
        days_data = [r['day'] for r in st.session_state.rows]
        rates_data = [r['rate'] / 100.0 for r in st.session_state.rows]
        a, b, r_sq = fit_retention_curve(days_data, rates_data)
        
        if a is not None:
            dnu_list = [daily_dnu] * (forecast_days + 1)
            forecast_results = forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days)
            # 将所有结果锁存，防止其他输入框变动触发重新拟合
            st.session_state.fitted_params = {
                'a': a, 'b': b, 'r_sq': r_sq, 
                'forecast': forecast_results, 
                'days': forecast_days
            }
        else:
            st.error("拟合失败，请检查数据")

with col2:
    st.header("📈 预测分析报告")
    
    # 检查是否有锁存的计算结果
    if st.session_state.fitted_params:
        p = st.session_state.fitted_params
        
        # 指标展示
        m1, m2, m3 = st.columns(3)
        m1.metric("拟合系数 a", f"{p['a']:.4f}")
        m2.metric("衰减系数 b", f"{p['b']:.4f}")
        m3.metric("拟合优度 R²", f"{p['r_sq']:.4f}")

        # 绘图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(p['forecast']))), 
            y=p['forecast'],
            mode='lines+markers',
            name='预测DAU',
            line=dict(color='#3498db', width=3),
            fill='tozeroy',
            hovertemplate="预测天数: %{x}<br>活跃用户数: %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            title=f"未来 {p['days']} 天 DAU 预测走势",
            xaxis_title="预测天数", yaxis_title="活跃用户数 (DAU)",
            hovermode="x unified", yaxis=dict(tickformat=",d") 
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧮 长期价值计算 (LT)")
        # 修改这里的 N 天不会触发重新拟合，只会重新计算求和
        lt_n = st.number_input("计算前 N 天的累计留存价值 (LT)", value=30, key="lt_val_input")
        lt_val = sum(get_retention_rate(d, p['a'], p['b']) for d in range(lt_n + 1))
        st.success(f"前 {lt_n} 天的累计留存价值 (LT) 为: {lt_val:.2f} 天")
    else:
        st.info("👈 请配置参数后点击『开始执行预测』。")
