import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("App用户活跃预测模型")

st.header("输入参数")
current_dau = st.number_input("当前活跃用户数 (DAU)", min_value=0, value=10000)
forecast_days = st.number_input("预测天数", min_value=1, max_value=365, value=30)

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

st.subheader("历史留存率 (%)")
retention_days = st.number_input("输入留存率的天数", min_value=1, max_value=30, value=7)
retention_rates = []
for i in range(retention_days):
    rate = st.number_input(f"第 {i+1} 天留存率 (%)", min_value=0.0, max_value=100.0, value=40.0/(i+1))
    retention_rates.append(rate / 100.0)

def forecast_dau(current_dau, dnu_list, retention_rates, forecast_days):
    dau_forecast = [current_dau]
    historical_dnu = [0] * len(retention_rates)
    for t in range(forecast_days):
        dau = dnu_list[t]
        for i in range(min(t + 1, len(retention_rates))):
            if t - i >= 0:
                dau += dnu_list[t - i] * retention_rates[i]
            else:
                dau += historical_dnu[i - (t + 1)] * retention_rates[i]
        dau_forecast.append(dau)
    return dau_forecast

if st.button("预测"):
    dau_forecast = forecast_dau(current_dau, dnu_list, retention_rates, forecast_days)
    st.header("预测结果")
    df_forecast = pd.DataFrame({
        "天数": range(forecast_days + 1),
        "活跃用户数 (DAU)": dau_forecast
    })
    st.write(df_forecast)
    st.subheader("DAU预测趋势")
    plt.figure(figsize=(10, 6))
    plt.plot(df_forecast["天数"], df_forecast["活跃用户数 (DAU)"], marker='o')
    plt.xlabel("天数")
    plt.ylabel("活跃用户数 (DAU)")
    plt.title("未来DAU预测")
    plt.grid(True)
    st.pyplot(plt)

with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nmatplotlib")
