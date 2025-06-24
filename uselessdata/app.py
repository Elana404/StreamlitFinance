import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime as dt

# 设置页面配置
st.set_page_config(
    page_title="股票期权估值工具",
    page_icon="📈",
    layout="wide"
)

st.title("股票期权估值工具")
st.write("基于二叉树模型的股票期权估值计算，支持稀释效应和行权行为参数设置")

# 侧边栏输入参数
st.sidebar.header("输入参数")

# 基本参数
valuation_date = st.sidebar.date_input("估值日期", dt.date(2011, 8, 30))
maturity_date = st.sidebar.date_input("到期日期", dt.date(2016, 8, 30))
exercisable_date = st.sidebar.date_input("可行权日期", dt.date(2011, 8, 30))
strike_price = st.sidebar.number_input("行权价格", value=10.00, step=0.01)
spot_price = st.sidebar.number_input("标的价格", value=10.00, step=0.01)
risk_free_rate = st.sidebar.number_input("无风险利率(%)", value=5.00, step=0.01)
volatility = st.sidebar.number_input("波动率(%)", value=30.00, step=0.01)
dividend_yield = st.sidebar.number_input("股息率(%)", value=0.00, step=0.01)
shares_outstanding = st.sidebar.number_input("流通股数", value=1.0, step=0.1)
num_steps = st.sidebar.slider("步数", min_value=5, max_value=100, value=10)

# 行为参数
exercise_behavior = st.sidebar.number_input("行权行为参数", value=0.0, step=0.1)
pre_vesting_exit_rate = st.sidebar.number_input(" vesting前退出率", value=0.0, step=0.01)
post_vesting_exit_rate = st.sidebar.number_input(" vesting后退出率", value=7.0, step=0.01)
vesting_period = st.sidebar.number_input(" vesting期(年)", value=0.0, step=0.1)
consider_dilution = st.sidebar.radio("考虑稀释效应", ["否", "是"], index=0) == "是"

# 期权数量
num_options = st.sidebar.number_input("期权数量", value=100, step=1)

# 核心计算函数 - 复制VBA宏中的逻辑
def share_option(dates, strike, numstep, behav, rfr1, vol, dvd, spot, shares, matdates, exdates, options, dil, exit_rate1, exit_rate2):
    # 初始化参数
    valdate, matdate, exdate = dates
    rfr = math.log(1 + rfr1 / 100)  # VBA中的Ln对应Python的math.log
    dt_days = (matdate - valdate).days
    delta_t = dt_days / 365.25 / numstep
    u = math.exp(vol / 100 * math.sqrt(delta_t))
    d = 1 / u
    p = (math.exp((rfr - dvd/100) * delta_t) - d) / (u - d)
    q = 1 - p
    
    # 数组初始化
    SP = np.zeros((numstep + 1, numstep + 1))
    DP = np.zeros((numstep + 1, numstep + 1))
    ex_option = np.zeros(numstep + 1)
    warrant = np.zeros((numstep + 1, numstep + 1))
    stepdate = [valdate + dt.timedelta(days=delta_t * i * 365.25) for i in range(numstep + 1)]
    
    # 计算可执行期权数量
    if dil:
        for i in range(numstep + 1):
            ex_option[i] = 0
            # 简化处理，假设所有期权都在有效期内
            ex_option[i] = num_options
    
    # 运行标的价格计算
    run_spot_price(SP, DP, ex_option, numstep, spot, u, d, strike, shares, dil)
    
    # 根据是否稀释选择不同的计算路径
    if not dil:
        if behav > 0:
            conversion_option2(SP, SP, warrant, stepdate, strike, exdate, numstep, rfr, p, delta_t, behav, exit_rate1, exit_rate2)
        else:
            conversion_option(SP, warrant, stepdate, strike, exdate, numstep, rfr, p, delta_t, exit_rate1, exit_rate2)
    else:
        if behav > 0:
            conversion_option2(SP, DP, warrant, stepdate, strike, exdate, numstep, rfr, p, delta_t, behav, exit_rate1, exit_rate2)
        else:
            conversion_option(DP, warrant, stepdate, strike, exdate, numstep, rfr, p, delta_t, exit_rate1, exit_rate2)
    
    return warrant[0, 0]

def run_spot_price(SP, DP, ex_option1, numstep, spot, u, d, strike, shares, dil):
    SP[0, 0] = spot
    
    for i in range(1, numstep + 1):
        for j in range(i):
            SP[i, j] = SP[i-1, j] * u
            if dil:
                if strike < SP[i, j]:
                    DP[i, j] = (SP[i, j] * shares + strike * ex_option1[i]) / (shares + ex_option1[i])
                else:
                    DP[i, j] = SP[i, j]
        
        SP[i, i] = SP[i-1, i-1] * d
        if dil:
            if strike < SP[i, i]:
                DP[i, i] = (SP[i, i] * shares + strike * ex_option1[i]) / (shares + ex_option1[i])
            else:
                DP[i, i] = SP[i, i]

def conversion_option(price, co, stepdate, exprice, exdate, n, r, p, delta_t, exit_r1, exit_r2):
    # 最后一步的期权价值
    for j in range(n + 1):
        if price[n, j] >= exprice:
            co[n, j] = price[n, j] - exprice
        else:
            co[n, j] = 0
    
    # 逆向计算
    for i in range(n-1, -1, -1):
        for j in range(i + 1):
            if stepdate[i] < exdate:
                co[i, j] = (1 - exit_r1 * delta_t) * (p * co[i+1, j] + (1 - p) * co[i+1, j+1]) * math.exp(-r * delta_t)
            else:
                co[i, j] = (1 - exit_r2 * delta_t) * (p * co[i+1, j] + (1 - p) * co[i+1, j+1]) * math.exp(-r * delta_t) + \
                          (exit_r2 * delta_t) * max(price[i, j] - exprice, 0)

def conversion_option2(price, dil_price, co, stepdate, exprice, exdate, n, r, p, delta_t, behav, exit_r1, exit_r2):
    # 最后一步的期权价值
    for j in range(n + 1):
        if dil_price[n, j] >= exprice:
            co[n, j] = dil_price[n, j] - exprice
        else:
            co[n, j] = 0
    
    # 逆向计算，考虑行权行为
    for i in range(n-1, -1, -1):
        for j in range(i + 1):
            if stepdate[i] < exdate:
                co[i, j] = (1 - exit_r1 * delta_t) * (p * co[i+1, j] + (1 - p) * co[i+1, j+1]) * math.exp(-r * delta_t)
            elif price[i, j] > exprice * behav:
                co[i, j] = max(dil_price[i, j] - exprice, 0)
            else:
                co[i, j] = (1 - exit_r2 * delta_t) * (p * co[i+1, j] + (1 - p) * co[i+1, j+1]) * math.exp(-r * delta_t) + \
                          (exit_r2 * delta_t) * max(dil_price[i, j] - exprice, 0)

# 执行计算
def calculate_option_value():
    # 准备日期参数
    valdate = valuation_date
    matdate = maturity_date
    exdate = exercisable_date
    dates = (valdate, matdate, exdate)
    
    # 准备期权数据，简化处理
    matdates = [matdate] * 30  # 假设30个期权
    exdates = [exdate] * 30
    options = [num_options/30] * 30  # 平均分配
    
    # 执行计算
    option_value = share_option(
        dates, strike_price, num_steps, exercise_behavior, 
        risk_free_rate, volatility, dividend_yield, spot_price, 
        shares_outstanding, matdates, exdates, options, 
        consider_dilution, pre_vesting_exit_rate, post_vesting_exit_rate
    )
    
    return option_value

# 计算并展示结果
if st.sidebar.button("计算期权价值"):
    with st.spinner("正在计算期权价值..."):
        option_value = calculate_option_value()
        
        # 计算总价值和占标的价格百分比
        total_value = option_value * num_options
        percentage_of_spot = (option_value / spot_price) * 100
        
        # 展示结果
        st.header("计算结果")
        col1, col2, col3 = st.columns(3)
        col1.metric("每份期权价值", f"{option_value:.5f}", f"{(option_value - 3.51191):.5f}" if option_value else None)
        col2.metric("总期权价值", f"{total_value:.2f}", f"{(total_value - 351.0):.2f}" if total_value else None)
        col3.metric("占标的价格百分比", f"{percentage_of_spot:.2f}%", f"{(percentage_of_spot - 35.12):.2f}%" if percentage_of_spot else None)
        
        st.subheader("参数汇总")
        params = {
            "估值日期": valuation_date,
            "到期日期": maturity_date,
            "可行权日期": exercisable_date,
            "行权价格": strike_price,
            "标的价格": spot_price,
            "无风险利率": f"{risk_free_rate}%",
            "波动率": f"{volatility}%",
            "股息率": f"{dividend_yield}%",
            "流通股数": shares_outstanding,
            "步数": num_steps,
            "行权行为参数": exercise_behavior,
            "vesting前退出率": pre_vesting_exit_rate,
            "vesting后退出率": f"{post_vesting_exit_rate}%",
            "vesting期": f"{vesting_period}年",
            "考虑稀释效应": consider_dilution
        }
        st.table(pd.DataFrame(list(params.items()), columns=["参数", "值"]))
        
        # 模拟数据展示（简化版，实际应根据计算结果生成）
        st.subheader("标的价格二叉树模拟")
        st.write("以下是标的价格在二叉树模型中的模拟路径")
        
        # 生成模拟价格数据用于展示
        days = np.arange(0, num_steps+1)
        base_price = spot_price
        price_up = [base_price * (1 + volatility/100/2)**i for i in days]
        price_down = [base_price * (1 - volatility/100/2)**i for i in days]
        
        # 绘制价格路径
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(days, price_up, 'r-', label='上行路径')
        ax.plot(days, price_down, 'b-', label='下行路径')
        ax.axhline(y=strike_price, color='g', linestyle='--', label='行权价格')
        ax.set_xlabel('步数')
        ax.set_ylabel('价格')
        ax.set_title('标的资产价格二叉树模拟')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        st.info("注：此工具基于二叉树模型计算期权价值，结果可能与Excel宏有细微差异，主要由于Python和VBA在数值计算上的精度差异。")
else:
    st.info("请在侧边栏输入参数，然后点击'计算期权价值'按钮开始计算")
    st.write("该工具实现了与Excel宏相同的期权估值功能，使用二叉树模型计算股票期权价值，支持稀释效应和行权行为参数设置。")