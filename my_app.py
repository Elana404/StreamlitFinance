import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import altair as alt

# =============================================================================
# 1. PYTHON LOGIC (Translated from VBA) - UNCHANGED
# =============================================================================

# (The Python calculation functions: conversion_option, conversion_option2, 
# run_spot_price, and share_option remain exactly the same as before.
# To save space, I will omit them here, but you should copy them into your file.)

def conversion_option(price, n, r, p, dt, exprice, step_dates, ex_date, exit_r1, exit_r2):
    co = np.zeros_like(price)
    co[n, :] = np.maximum(price[n, :] - exprice, 0)
    for i in range(n - 1, -1, -1):
        expected_value = (p * co[i + 1, :i+1] + (1 - p) * co[i + 1, 1:i+2]) * np.exp(-r * dt)
        if step_dates[i] < ex_date:
            co[i, :i+1] = (1 - exit_r1 * dt) * expected_value
        else:
            forfeiture_value = np.maximum(price[i, :i+1] - exprice, 0)
            co[i, :i+1] = (1 - exit_r2 * dt) * expected_value + (exit_r2 * dt) * forfeiture_value
    return co

def conversion_option2(price, dil_price, n, r, p, dt, exprice, step_dates, ex_date, behav, exit_r1, exit_r2):
    co = np.zeros_like(price)
    co[n, :] = np.maximum(dil_price[n, :] - exprice, 0)
    for i in range(n - 1, -1, -1):
        expected_value = (p * co[i + 1, :i+1] + (1 - p) * co[i + 1, 1:i+2]) * np.exp(-r * dt)
        if step_dates[i] < ex_date:
            co[i, :i+1] = (1 - exit_r1 * dt) * expected_value
        else:
            is_early_exercise = (price[i, :i+1] > exprice * behav) & (behav > 0)
            # Must loop because exercise decision is node-specific
            for j in range(i + 1):
                if is_early_exercise[j]:
                    co[i, j] = np.maximum(dil_price[i, j] - exprice, 0)
                else:
                    forfeiture_value = np.maximum(dil_price[i, j] - exprice, 0)
                    co[i, j] = (1 - exit_r2 * dt) * (p * co[i + 1, j] + (1 - p) * co[i + 1, j + 1]) * np.exp(-r * dt) + \
                               (exit_r2 * dt) * forfeiture_value
    return co

def run_spot_price(numstep, spot, u, d, strike, shares, dil, num_options_in_lot):
    sp = np.zeros((numstep + 1, numstep + 1))
    dp = np.zeros((numstep + 1, numstep + 1))
    sp[0, 0] = spot
    for i in range(1, numstep + 1):
        sp[i, :i] = sp[i-1, :i] * u
        sp[i, i] = sp[i-1, i-1] * d
    if not dil:
        return sp, sp
    exercisable_options = num_options_in_lot
    for i in range(numstep + 1):
        for j in range(i + 1):
            if sp[i, j] > strike:
                dp[i, j] = (sp[i, j] * shares + strike * exercisable_options) / (shares + exercisable_options)
            else:
                dp[i, j] = sp[i, j]
    return sp, dp

def share_option(val_date, mat_date, ex_date, strike, spot, numstep, behav, rfr1,
                 vol, dvd, shares, num_options_in_lot, dil, exit_rate1, exit_rate2):
    if not all(isinstance(d, date) for d in [val_date, mat_date, ex_date]):
        return "Error: Invalid date provided."
    rfr = np.log(1 + rfr1)
    life_in_days = (mat_date - val_date).days
    if life_in_days <= 0 or numstep <= 0:
        return 0.0
    dt = (life_in_days / 365.25) / numstep
    u = np.exp(vol * np.sqrt(dt))
    if u == 1: return 0.0
    d = 1 / u
    p = (np.exp((rfr - dvd) * dt) - d) / (u - d)
    step_dates = [val_date + timedelta(days=365.25 * i * dt) for i in range(numstep + 1)]
    sp_tree, dp_tree = run_spot_price(numstep, spot, u, d, strike, shares, dil, num_options_in_lot)
    if behav > 0:
        warrant_tree = conversion_option2(sp_tree, dp_tree, numstep, rfr, p, dt, strike,
                                          step_dates, ex_date, behav, exit_rate1, exit_rate2)
    else:
        price_tree_to_use = dp_tree if dil else sp_tree
        warrant_tree = conversion_option(price_tree_to_use, numstep, rfr, p, dt, strike,
                                         step_dates, ex_date, exit_rate1, exit_rate2)
    return warrant_tree[0, 0]


# =============================================================================
# 2. STREAMLIT USER INTERFACE
# =============================================================================
st.set_page_config(layout="wide")

# Initialize session state for storing results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'calculation_mode' not in st.session_state:
    st.session_state.calculation_mode = "Single Company/Lot Valuation"

# --- SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Controls")
    
    mode = st.radio(
        "Select Calculation Mode",
        ("Single Company/Lot Valuation", "Batch Calculation from Table"),
        key="calculation_mode"
    )

    if mode == "Single Company/Lot Valuation":
        st.header("Global Parameters")
        decimal_digits = st.number_input("Decimal Digits for Option Values", min_value=0, max_value=10, value=5, step=1)
        rfr1 = st.number_input("Risk Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
        vol = st.number_input("Volatility (%)", min_value=0.0, value=30.0, step=1.0) / 100
        dvd = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
        shares = st.number_input("Shares Outstanding", min_value=1, value=1_000_000, step=1000)
        dil = st.radio("Consider Dilution?", ("NO", "YES"), index=0) == "YES"

        st.header("Option Lot Parameters")
        val_date = st.date_input("Valuation Date", value=datetime(2011, 8, 30).date())
        mat_date = st.date_input("Maturity Date", value=datetime(2016, 8, 30).date())
        ex_date = st.date_input("Exercisable Date", value=datetime(2012, 8, 30).date())
        strike = st.number_input("Exercise Price", min_value=0.0, value=10.0, step=0.1)
        spot = st.number_input("Spot Price", min_value=0.0, value=10.0, step=0.1)
        num_options_in_lot = st.number_input("No. of Share Options", min_value=0, value=100, step=10)
        numstep = st.number_input("No. of Steps", min_value=1, value=10, step=1)
        behav = st.number_input("Exercise Behaviour Multiplier (%)", min_value=0.0, value=220.0, step=0.1) / 100
        exit_rate1 = st.number_input("Pre-vesting Exit Rate (%)", min_value=0.0, value=0.0, step=0.1) / 100
        exit_rate2 = st.number_input("Post-vesting Exit Rate (%)", min_value=0.0, value=7.0, step=0.1) / 100

        params = locals().copy() # Capture all local variables as parameters
        calculate_button = st.button("Calculate", type="primary", use_container_width=True)

    else: # Batch Calculation from Table
        st.header("Global Parameters")
        st.markdown("Parameters here apply to all lots in the main table.")
        decimal_digits = st.number_input("Decimal Digits for Option Values", min_value=0, max_value=10, value=5, step=1)
        rfr1 = st.number_input("Risk Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
        vol = st.number_input("Volatility (%)", min_value=0.0, value=30.0, step=1.0) / 100
        dvd = st.number_input("Dividend Yield (%)", min_value=0.0, value=0.0, step=0.1) / 100
        shares = st.number_input("Shares Outstanding", min_value=1, value=1_000_000, step=1000)
        dil = st.radio("Consider Dilution for All Lots?", ("NO", "YES"), index=0) == "YES"
        calculate_button = st.button("Calculate All Lots", type="primary", use_container_width=True)


# --- MAIN PANEL FOR DATA INPUT & RESULTS ---
st.title("📈 Share Option Valuation")

if mode == "Batch Calculation from Table":
    st.header("📄 Batch Input Table")
    st.markdown("Edit the data for each lot below. Dates should be YYYY-MM-DD. You can add or remove lots (columns).")

    # Create a DataFrame with the same structure as the Excel file's tan fields
    initial_data = {
        'Lot 1': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 0, 0.0, 0.0, 10.0, 100],
        'Lot 2': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 0, 0.0, 7.0, 10.0, 100],
        'Lot 3': ['2011-08-30', '2016-08-30', '2012-08-30', 10.0, 10, 0, 0.0, 7.0, 10.0, 100],
        'Lot 4': ['2011-08-30', '2016-08-30', '2012-08-30', 10.0, 10, 220, 0.0, 7.0, 10.0, 100],
        'Lot 5': ['2011-08-30', '2016-08-30', '2011-08-30', 10.0, 10, 220, 0.0, 0.0, 10.0, 100],
    }
    index_labels = [
        "Valuation Date", "Maturity Date", "Exercisable Date", "Exercise Price",
        "No. of Steps", "Exercise Behaviour Multiplier (%)", "Pre-vesting Exit Rate (%)",
        "Post-vesting Exit Rate (%)", "Spot Price", "No. of Share Options"
    ]
    df_input = pd.DataFrame(initial_data, index=index_labels)

    edited_df = st.data_editor(
        df_input,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
else:
    st.info("⬅️ Set parameters in the sidebar and click 'Calculate'.")


# --- CALCULATION LOGIC ---
if calculate_button:
    if mode == "Single Company/Lot Valuation":
        try:
            # We need to pass only the arguments required by the function
            required_params = {k: v for k, v in params.items() if k in share_option.__code__.co_varnames}
            result = share_option(**required_params)
            st.session_state.results = {'value': result, 'params': params}
        except Exception as e:
            st.error(f"An error occurred during calculation: {e}")
            st.session_state.results = None

    else: # Batch Mode
        results_list = []
        has_error = False
        for lot_name in edited_df.columns:
            try:
                p = edited_df[lot_name]
                option_value = share_option(
                    val_date=datetime.strptime(str(p["Valuation Date"]), '%Y-%m-%d').date(),
                    mat_date=datetime.strptime(str(p["Maturity Date"]), '%Y-%m-%d').date(),
                    ex_date=datetime.strptime(str(p["Exercisable Date"]), '%Y-%m-%d').date(),
                    strike=float(p["Exercise Price"]),
                    spot=float(p["Spot Price"]),
                    numstep=int(p["No. of Steps"]),
                    behav=float(p["Exercise Behaviour Multiplier (%)"]) / 100,
                    num_options_in_lot=float(p["No. of Share Options"]),
                    exit_rate1=float(p["Pre-vesting Exit Rate (%)"]) / 100,
                    exit_rate2=float(p["Post-vesting Exit Rate (%)"]) / 100,
                    # Global parameters from sidebar
                    rfr1=rfr1, vol=vol, dvd=dvd, dil=dil, shares=shares
                )
                if isinstance(option_value, str):
                    st.error(f"Error in {lot_name}: {option_value}")
                    has_error = True
                else:
                    total_value = option_value * float(p["No. of Share Options"])
                    results_list.append({'Lot': lot_name, 'Per Option Value': option_value, 'Total Value': total_value})
            except Exception as e:
                st.error(f"An error occurred while processing {lot_name}: {e}")
                has_error = True

        if not has_error and results_list:
            st.session_state.results = pd.DataFrame(results_list)
        else:
            st.session_state.results = None

# --- DISPLAY RESULTS AND CHARTS ---
if st.session_state.results is not None:
    st.header("📊 Results & Analysis")

    # --- Display for Single Mode ---
    if st.session_state.calculation_mode == "Single Company/Lot Valuation" and isinstance(st.session_state.results, dict):
        res = st.session_state.results
        p = res['params']
        digits = p.get('decimal_digits', 5)

        st.subheader("Calculation Outcome")
        col1, col2 = st.columns(2)
        col1.metric("Per Option Value", f"{res['value']:.{digits}f}")
        total_val = res['value'] * p['num_options_in_lot']
        col2.metric("Total Value for Lot", f"${total_val:,.2f}")

        st.subheader("Sensitivity Analysis (vs. Volatility)")
        sens_data = []
        base_vol = p['vol']
        # Create a copy of parameters for modification
        sens_params = {k: v for k, v in p.items() if k in share_option.__code__.co_varnames}
        for v_sens in np.linspace(max(0.01, base_vol * 0.5), base_vol * 1.5, 20):
            sens_params['vol'] = v_sens
            sens_data.append({
                'Volatility': v_sens,
                'Option Value': share_option(**sens_params)
            })
        df_sens = pd.DataFrame(sens_data)

        chart = alt.Chart(df_sens).mark_line(point=True).encode(
            x=alt.X('Volatility:Q', axis=alt.Axis(format='%')),
            y=alt.Y('Option Value:Q', axis=alt.Axis(format=f'$,.{digits}f')),
            tooltip=[alt.Tooltip('Volatility', format='.2%'), alt.Tooltip('Option Value', format=f'.{digits}f')]
        ).properties(
            title='Option Value vs. Volatility'
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    # --- Display for Batch Mode ---
    elif st.session_state.calculation_mode == "Batch Calculation from Table" and isinstance(st.session_state.results, pd.DataFrame):
        df_res = st.session_state.results
        digits = st.session_state.get('decimal_digits', 5) # Retrieve from session state or default

        st.subheader("Results Table")
        st.dataframe(
            df_res.style.format({
                'Per Option Value': f'{{:.{digits}f}}',
                'Total Value': '${:,.2f}'
            }),
            use_container_width=True
        )

        total_sum = df_res['Total Value'].sum()
        st.metric(label="Grand Total Value (All Lots)", value=f"${total_sum:,.2f}")

        st.subheader("Analysis Charts")
        # Bar chart for comparing values
        bar_chart = alt.Chart(df_res).mark_bar().encode(
            x=alt.X('Lot:N', sort=None, title='Lot Number'),
            y=alt.Y('Per Option Value:Q', title='Per Option Value ($)'),
            color=alt.Color('Lot:N', legend=None),
            tooltip=['Lot', alt.Tooltip('Per Option Value', format=f'.{digits}f')]
        ).properties(
            title='Per Option Value Comparison by Lot'
        ).interactive()
        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Benchmark Comparison (from Excel)")
        st.markdown("`Values: Lot 1=3.51191, Lot 2=3.05534, Lot 3=3.30977, Lot 4=3.17827, Lot 5=3.3545`")