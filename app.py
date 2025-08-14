import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("App用户活跃预测模型（更新版）")

# 使用列布局：左边输入，中部图表，右边结论
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.header("输入参数")
    current_dau = st.number_input("当前活跃用户数 (DAU)", min_value=0, value=10000)
    forecast_days = st.number_input("预测天数", min_value=1, max_value=365, value=30)
    churn_rate = st.number_input("老用户每日流失率 (%)", min_value=0.0, max_value=100.0, value=1.0) / 100.0

    st.subheader("每日新增用户 (DNU)")
    use_fixed_dnu = st.checkbox("使用固定每日新增用户", value=True)
    if use_fixed_dnu:
        daily_dnu = st.number_input("每日新增用户数", min_value=0, value=500)
        dnu_list = [daily_dnu] * forecast_days
    else:
        dnu_list = []
        for i in range(forecast_days):
            dnu = st.number_input(f"第 {i+1} 天新增用户数", min_value=0, value=500)
            dnu_list.append(dnu)

    st.subheader("留存率输入（可间断）")
    if 'retention_points' not in st.session_state:
        st.session_state.retention_points = []

    def add_retention_point():
        st.session_state.retention_points.append({'day': 1, 'rate': 50.0})

    st.button("添加留存点", on_click=add_retention_point)

    retention_days = []
    retention_rates = []
    for idx, point in enumerate(st.session_state.retention_points):
        point['day'] = st.number_input(f"留存点 {idx+1} - 天数", min_value=1, value=point.get('day', 1), key=f"day_{idx}")
        point['rate'] = st.number_input(f"留存点 {idx+1} - 留存率 (%)", min_value=0.0, max_value=100.0, value=point.get('rate', 50.0), key=f"rate_{idx}") / 100.0
        retention_days.append(point['day'])
        retention_rates.append(point['rate'])

    # 确保天数唯一并排序
    if retention_days:
        sorted_indices = np.argsort(retention_days)
        retention_days = [retention_days[i] for i in sorted_indices]
        retention_rates = [retention_rates[i] for i in sorted_indices]

with col3:
    st.header("结论与拟合结果")

def fit_retention_curve(days, rates):
    if len(days) < 2:
        return None, None, 0.0
    # 假设幂律拟合: ret = a * day ** (-b)
    log_days = np.log(days)
    log_rates = np.log(rates)
    b, log_a = np.polyfit(log_days, log_rates, 1)
    a = np.exp(log_a)
    b = -b  # 因为 (-b)
    
    # 计算拟合值
    fitted_rates = a * np.power(days, -b)
    
    # R² 计算
    ss_res = np.sum((rates - fitted_rates) ** 2)
    ss_tot = np.sum((rates - np.mean(rates)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return a, b, r_squared

def get_retention_rate(day, a, b):
    if a is None or b is None:
        return 0.0
    return a * (day ** (-b)) if day > 0 else 0.0

def forecast_dau(current_dau, dnu_list, retention_func, churn_rate, forecast_days):
    dau_forecast = [current_dau]
    # cohort_dnu: 假设未来cohort的dnu_list, 历史cohort简化为current_dau作为一个整体老cohort
    old_dau = current_dau  # 初始老用户
    cohort_contributions = [[] for _ in range(forecast_days + 1)]  # 每个cohort的贡献
    
    for t in range(forecast_days):
        # 更新老用户：应用流失率
        old_dau *= (1 - churn_rate)
        
        # 新cohort: dnu_list[t]
        dau = dnu_list[t] + old_dau  # 当天新 + 老
        
        # 添加之前cohort的留存贡献
        for prev_t in range(t):
            retention_day = t - prev_t
            dau += dnu_list[prev_t] * retention_func(retention_day)
        
        dau_forecast.append(dau)
    
    return dau_forecast

if st.button("预测"):
    a, b, r_squared = fit_retention_curve(retention_days, retention_rates)
    
    def retention_func(day):
        return get_retention_rate(day, a, b)
    
    dau_forecast = forecast_dau(current_dau, dnu_list, retention_func, churn_rate, forecast_days)
    
    df_forecast = pd.DataFrame({
        "天数": range(forecast_days + 1),
        "活跃用户数 (DAU)": dau_forecast
    })
    
    with col2:
        st.header("预测结果")
        st.write(df_forecast)
        st.subheader("DAU预测趋势")
        plt.figure(figsize=(10, 6))
        plt.plot(df_forecast["天数"], df_forecast["活跃用户数 (DAU)"], marker='o')
        plt.xlabel("天数")
        plt.ylabel("活跃用户数 (DAU)")
        plt.title("未来DAU预测")
        plt.grid(True)
        st.pyplot(plt)
    
    with col3:
        if a is not None and b is not None:
            st.write(f"拟合留存公式: retention = {a:.4f} * day ^ (-{b:.4f})")
            st.write(f"R² 值: {r_squared:.4f}")
        else:
            st.write("至少需要两个留存点进行拟合。")

with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nmatplotlib")