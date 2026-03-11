import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 核心算法部分 ---

def power_law(x, a, b):
    """定义幂函数模型: y = a * x^(-b)"""
    return a * np.power(x, -b)

def fit_retention_curve(days, rates):
    """使用非线性最小二乘法进行拟合"""
    if len(days) < 2:
        return None, None, 0.0
    
    # 转换为 numpy 数组确保计算性能
    days_arr = np.array(days)
    rates_arr = np.array(rates)
    
    try:
        # p0: 初始猜测值 [a=0.5, b=0.5]
        # bounds: 限制 a 在 [0, 1] 之间（留存率上限100%），b > 0（衰减方向）
        popt, pcov = curve_fit(power_law, days_arr, rates_arr, p0=[0.5, 0.5], bounds=(0, [1.0, np.inf]))
        a, b = popt
        
        # 计算拟合优度 R²
        fitted_rates = power_law(days_arr, a, b)
        ss_res = np.sum((rates_arr - fitted_rates) ** 2)
        ss_tot = np.sum((rates_arr - np.mean(rates_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return a, b, r_squared
    except Exception as e:
        st.error(f"非线性拟合失败，请检查输入数据是否合理: {e}")
        return None, None, 0.0

def get_retention_rate(day, a, b):
    """根据拟合参数获取特定天数的留存率"""
    if a is None or b is None:
        return 0.0
    if day == 0:
        return 1.0  # 定义 D0 留存始终为 100%
    return a * (day ** (-b)) if day > 0 else 0.0

def forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days):
    """DAU 预测核心逻辑"""
    dau_forecast = [current_dau]
    old_dau = current_dau
    
    for t in range(1, forecast_days + 1):
        # 1. 存量老用户自然流失
        old_dau *= (1 - churn_rate)
        
        # 2. 预测期间新增用户(DNU)在第 t 天的留存贡献
        new_user_contribution = 0
        for prev_t in range(t):
            # 距离新增那一刻过去了多少天
            retention_day = t - (prev_t + 1)
            # 获取对应的留存率并累加
            new_user_contribution += dnu_list[prev_t] * get_retention_rate(retention_day, a, b)
            
        current_day_dau = old_dau + new_user_contribution
        dau_forecast.append(current_day_dau)
        
    return dau_forecast

# --- UI 界面部分 ---

st.set_page_config(page_title="App用户活跃预测-科学版", layout="wide")

# 自定义CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #3498db; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📱 App用户活跃预测模型（Scipy科学拟合版）")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("📊 参数输入")
    with st.expander("基础配置", expanded=True):
        current_dau = st.number_input("当前 DAU", value=10000, step=100)
        forecast_days = st.slider("预测天数", 7, 365, 30)
        churn_rate = st.number_input("老用户每日流失率 (%)", value=1.0, step=0.1) / 100.0

    with st.expander("DNU (新增) 配置", expanded=True):
        daily_dnu = st.number_input("每日平均新增", value=500, step=50)
        dnu_list = [daily_dnu] * (forecast_days + 1)

    st.header("🔄 留存点输入")
    if 'rows' not in st.session_state:
        st.session_state.rows = [{'day': 1, 'rate': 50.0}, {'day': 7, 'rate': 30.0}]

    def add_row():
        st.session_state.rows.append({'day': 30, 'rate': 15.0})

    for i, row in enumerate(st.session_state.rows):
        c1, c2 = st.columns([1, 1])
        st.session_state.rows[i]['day'] = c1.number_input(f"天数", value=row['day'], key=f"d_{i}")
        st.session_state.rows[i]['rate'] = c2.number_input(f"留存%", value=row['rate'], key=f"r_{i}")

    st.button("➕ 添加留存观测点", on_click=add_row)

with col2:
    st.header("📈 预测分析报告")
    
    # 提取数据并拟合
    days_data = [r['day'] for r in st.session_state.rows]
    rates_data = [r['rate'] / 100.0 for r in st.session_state.rows]
    
    a, b, r_sq = fit_retention_curve(days_data, rates_data)
    
    if a is not None:
        # 计算预测
        forecast_results = forecast_dau(current_dau, dnu_list, a, b, churn_rate, forecast_days)
        
        # 布局展示
        m1, m2, m3 = st.columns(3)
        m1.metric("拟合系数 a (D1趋势)", f"{a:.4f}")
        m2.metric("衰减系数 b", f"{b:.4f}")
        m3.metric("拟合优度 R²", f"{r_sq:.4f}")

        # 图表展示
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(forecast_results)), forecast_results, marker='o', linestyle='-', color='#3498db', label="预测DAU")
        ax.fill_between(range(len(forecast_results)), forecast_results, color='#3498db', alpha=0.1)
        ax.set_title(f"未来 {forecast_days} 天 DAU 预测走势")
        ax.set_xlabel("预测天数")
        ax.set_ylabel("DAU")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # LT 计算
        st.subheader("🧮 长期价值计算 (LTV/LT)")
        lt_n = st.number_input("计算前 N 天的留存累加值 (LT)", value=30)
        lt_val = sum(get_retention_rate(d, a, b) for d in range(lt_n + 1))
        st.success(f"前 {lt_n} 天的累计留存价值 (LT) 为: {lt_val:.2f} 天")
        
        # 数据详情
        with st.expander("查看预测数据详情"):
            df = pd.DataFrame({
                "预测天数": range(len(forecast_results)),
                "预计DAU": [int(x) for x in forecast_results]
            })
            st.dataframe(df.T)
    else:
        st.warning("请至少输入两个有效的留存点以开始科学拟合。")

# 依赖声明文件生成
with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nmatplotlib\nscipy")
