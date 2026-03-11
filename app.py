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
        # 使用 scipy 进行非线性拟合
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

st.set_page_config(page_title="App买量 ROAS 预测", layout="wide")
st.title("📱 App买量 ROAS 预测模型 (考虑渠道分成)")

if 'calculated' not in st.session_state:
    st.session_state.calculated = False

col1, col2 = st.columns([1, 3])

with col1:
    st.header("📊 参数输入")
    with st.expander("1. 活跃与流失配置", expanded=True):
        current_dau = st.number_input("当前 DAU", value=10000)
        forecast_days = st.slider("预测天数", 7, 90, 30)
        churn_rate = st.number_input("老用户每日流失率 (%)", value=1.0) / 100.0
        daily_dnu = st.number_input("每日平均新增 (DNU)", value=500)

    with st.expander("2. 买量与营收配置 (ROAS)", expanded=True):
        arpu = st.number_input("原始 ARPU (活跃用户日均产出)", value=2.0)
        # 优化点：手动输入渠道分成比例
        channel_share_input = st.number_input("渠道分成比例 (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        channel_share = channel_share_input / 100.0
        cac = st.number_input("买量单价 (CAC)", value=10.0)
    
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
    run_calc = st.button("🚀 开始执行预测", type="primary", use_container_width=True)
    if run_calc:
        st.session_state.calculated = True

with col2:
    st.header("📈 ROAS 分析报告")
    
    if st.session_state.calculated:
        days_data = [r['day'] for r in st.session_state.rows]
        rates_data = [r['rate'] / 100.0 for r in st.session_state.rows]
        
        a, b, r_sq = fit_retention_curve(days_data, rates_data)
        
        if a is not None:
            # 基础预测
            dnu_list = [daily_dnu] * (forecast_days + 1)
            forecast_results = forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days)
            
            # --- ROAS 核心计算 ---
            # 结算后 ARPU = 原始 ARPU * (1 - 分成比例)
            net_arpu = arpu * (1 - channel_share) 
            lt_n = st.number_input("ROAS 观察周期 (天)", value=30, help="计算该周期内的累计净收入是否回本")
            lt_val = sum(get_retention_rate(d, a, b) for d in range(lt_n + 1))
            # 结算后 LTV = 累计留存天数 * 结算后 ARPU
            net_ltv = lt_val * net_arpu 
            # ROAS = 结算后 LTV / 买量成本
            roas = (net_ltv / cac) * 100 if cac > 0 else 0 

            # 指标展示
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("结算后日均 ARPU", f"￥{net_arpu:.2f}")
            m2.metric(f"D{lt_n} 累计 LT", f"{lt_val:.2f} 天")
            m3.metric(f"D{lt_n} 结算后 LTV", f"￥{net_ltv:.2f}")
            m4.metric(f"D{lt_n} 买量 ROAS", f"{roas:.1f}%", 
                      delta=f"{roas-100:.1f}%" if roas>0 else None, 
                      delta_color="normal" if roas >= 100 else "inverse")

            # 诊断意见
            if roas >= 100:
                st.success(f"🎊 渠道表现优秀：在第 {lt_n} 天已实现回本！当前 ROAS 为 {roas:.1f}%。")
            else:
                st.warning(f"🚨 渠道尚未回本：在第 {lt_n} 天 ROAS 为 {roas:.1f}%。距离回本还差 {100-roas:.1f}%。")

            # Plotly 绘图
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

            with st.expander("🔬 查看数学拟合详情"):
                st.write(f"拟合幂函数公式: $Retention = {a:.4f} \cdot day^{{-{b:.4f}}}$")
                st.write(f"拟合优度 $R^2$: {r_sq:.4f}")
        else:
            st.error("拟合失败。请确保至少输入两个不同的留存观测点（如 D1 和 D7）。")
    else:
        st.info("👈 请在左侧配置参数后点击『开始执行预测』按钮。")

# 依赖文件自动生成
with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nscipy\nplotly")
