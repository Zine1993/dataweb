import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def fit_retention_curve(days, rates):
    if len(days) < 2:
        return None, None, 0.0
    log_days = np.log(days)
    log_rates = np.log(rates)
    b, log_a = np.polyfit(log_days, log_rates, 1)
    a = np.exp(log_a)
    b = -b
    fitted_rates = a * np.power(days, -b)
    ss_res = np.sum((rates - fitted_rates) ** 2)
    ss_tot = np.sum((rates - np.mean(rates)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return a, b, r_squared

def get_retention_rate(day, a, b):
    if a is None or b is None:
        return 0.0
    if day == 0:
        return 1.0  # D0留存=1
    return a * (day ** (-b)) if day > 0 else 0.0

def forecast_dau(current_dau, dnu_list, retention_func, churn_rate, forecast_days):
    dau_forecast = [current_dau]
    old_dau = current_dau
    for t in range(forecast_days):
        old_dau *= (1 - churn_rate)
        dau = dnu_list[t] + old_dau
        for prev_t in range(t):
            retention_day = t - prev_t
            dau += dnu_list[prev_t] * retention_func(retention_day)
        dau_forecast.append(dau)
    return dau_forecast

# 自定义CSS美化
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #f0f4f8, #e0e8f0);
    }
    .stButton > button {
        background: linear-gradient(to right, #3498db, #2980b9);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #2980b9, #3498db);
        transform: scale(1.05);
    }
    .stNumberInput > div > input {
        border-radius: 10px;
        border: 2px solid #bdc3c7;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stNumberInput > div > input:hover {
        border-color: #3498db;
        box-shadow: 0 6px 12px rgba(52,152,219,0.2);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    .stInfo {
        background-color: #d9edf7;
        border: 1px solid #bce8f1;
        color: #31708f;
        border-radius: 10px;
        padding: 10px;
    }
    .chart-title {
        background: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 App用户活跃预测模型（美化焕新版）")

# 两列布局：左边输入，右边大输出
col1, col2 = st.columns([1, 4])

with col1:
    st.header("📊 输入参数")
    current_dau = st.number_input("当前活跃用户数 (DAU)", min_value=0, value=10000, key="current_dau")
    forecast_days = st.number_input("预测天数", min_value=1, max_value=365, value=30, key="forecast_days")
    churn_rate = st.number_input("老用户每日流失率 (%)", min_value=0.0, max_value=100.0, value=1.0, key="churn_rate") / 100.0

    st.subheader("📈 每日新增用户 (DNU)")
    use_fixed_dnu = st.checkbox("使用固定每日新增用户", value=True, key="use_fixed_dnu")
    if use_fixed_dnu:
        daily_dnu = st.number_input("每日新增用户数", min_value=0, value=500, key="daily_dnu")
        dnu_list = [daily_dnu] * forecast_days
    else:
        dnu_list = []
        for i in range(forecast_days):
            dnu = st.number_input(f"第 {i+1} 天新增用户数", min_value=0, value=500, key=f"dnu_{i}")
            dnu_list.append(dnu)

    st.subheader("🔄 留存率输入（可间断）")
    # 初始化 session_state
    if 'retention_points' not in st.session_state:
        st.session_state.retention_points = []
    if 'temp_retention_points' not in st.session_state:
        st.session_state.temp_retention_points = []

    def add_retention_point():
        st.session_state.temp_retention_points.append({'day': 1, 'rate': 50.0 / 100.0})

    st.button("添加留存点", on_click=add_retention_point, key="add_retention")

    # 临时存储输入
    for idx in range(len(st.session_state.temp_retention_points)):
        point = st.session_state.temp_retention_points[idx]
        col_a, col_b, col_c = st.columns([1, 1, 0.5])
        with col_a:
            new_day = st.number_input(f"留存点 {idx+1} - 天数", min_value=1, value=point.get('day', 1), key=f"day_{idx}_{st.session_state.get('input_version', 0)}")
        with col_b:
            new_rate_percent = st.number_input(f"留存点 {idx+1} - 留存率 (%)", min_value=0.0, max_value=100.0, value=point.get('rate', 0.5) * 100.0, key=f"rate_percent_{idx}_{st.session_state.get('input_version', 0)}")
            new_rate = new_rate_percent / 100.0
        with col_c:
            if st.button("移除", key=f"remove_{idx}_{st.session_state.get('input_version', 0)}"):
                st.session_state.temp_retention_points.pop(idx)
                st.rerun()

        st.session_state.temp_retention_points[idx]['day'] = new_day
        st.session_state.temp_retention_points[idx]['rate'] = new_rate

    # 保存按钮
    if st.button("保存留存点", key="save_retention"):
        st.session_state.retention_points = [dict(p) for p in st.session_state.temp_retention_points]
        st.session_state.input_version = st.session_state.get('input_version', 0) + 1
        st.success("留存点保存成功！")
        st.rerun()

    # 显示调试信息
    with st.expander("查看留存点数据"):
        st.write("已保存的留存点：")
        st.write(st.session_state.retention_points)
        st.write("当前输入（未保存）：")
        st.write(st.session_state.temp_retention_points)

    # 准备数据
    retention_days = [point['day'] for point in st.session_state.retention_points]
    retention_rates = [point['rate'] for point in st.session_state.retention_points]
    if retention_days:
        sorted_indices = np.argsort(retention_days)
        retention_days = [retention_days[i] for i in sorted_indices]
        retention_rates = [retention_rates[i] for i in sorted_indices]
        st.info("提示：留存点已按天数排序。")

# 右边大输出区
with col2:
    st.header("📈 预测结果与分析")
    if st.button("🔍 预测", key="forecast_button"):
        if not retention_days or not retention_rates:
            st.error("请至少保存一个留存点以进行预测。")
        else:
            a, b, r_squared = fit_retention_curve(retention_days, retention_rates)
            
            def retention_func(day):
                return get_retention_rate(day, a, b)
            
            dau_forecast = forecast_dau(current_dau, dnu_list, retention_func, churn_rate, forecast_days)
            
            df_forecast = pd.DataFrame({
                "天数": range(forecast_days + 1),
                "活跃用户数 (DAU)": dau_forecast
            })
            
            st.dataframe(df_forecast.style.format({"活跃用户数 (DAU)": "{:.0f}"}).set_properties(**{'border': '1px solid #ddd', 'padding': '8px'}))
            st.subheader("DAU预测趋势")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df_forecast["天数"], df_forecast["活跃用户数 (DAU)"], marker='o', color='#3498db', linewidth=2)
            ax.set_xlabel("天数")
            ax.set_ylabel("活跃用户数 (DAU)")
            ax.set_title("未来DAU预测", pad=15)
            ax.grid(True, linestyle='--', alpha=0.7)
            st.markdown('<div class="chart-title">DAU趋势图</div>', unsafe_allow_html=True)
            st.pyplot(fig)

    if 'a' in locals() and 'b' in locals() and a is not None and b is not None:
        st.write(f"拟合留存公式: retention = {a:.4f} * day ^ (-{b:.4f})")
        st.write(f"R² 值: {r_squared:.4f}")
    else:
        st.write("至少需要两个留存点进行拟合。")

    st.subheader("🧮 计算LT值（留存累加，包括D0=1）")
    lt_n = st.number_input("输入n天", min_value=1, value=30, key="lt_n")
    if st.button("计算LT", key="calc_lt"):
        a, b, _ = fit_retention_curve(retention_days, retention_rates)
        if a is not None and b is not None:
            lt_value = sum(get_retention_rate(day, a, b) for day in range(0, lt_n + 1))
            st.success(f"n={lt_n} 天的留存累加值 (包括D0=1): {lt_value:.4f}")
        else:
            st.warning("请先保存至少两个留存点并预测以拟合公式。")

with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nmatplotlib")