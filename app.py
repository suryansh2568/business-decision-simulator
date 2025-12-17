import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import itertools

# 1) Setup
st.set_page_config(page_title="Business Strategy Simulator", layout="wide")

# 2) Syntethic Data Generation
@st.cache_data
def generate_data(industry, n_rows=2000):
    np.random.seed(42)
    
    # 3) Defining Business Physics
    if industry == "Tech (Gadgets)":
        base_price = 800
        base_demand = 500
        elasticity = -2.0
        marketing_power = 50
    elif industry == "Fashion (Luxury)":
        base_price = 1200
        base_demand = 150
        elasticity = -0.6
        marketing_power = 100
    elif industry == "FMCG (Food)":
        base_price = 4          
        base_demand = 15000     
        elasticity = -6.5       
        marketing_power = 15    
    elif industry == "FMCG (Cosmetics)":
        base_price = 25         
        base_demand = 2000      
        elasticity = -1.2       
        marketing_power = 90    
    elif industry == "FMCG (Personal Care)":
        base_price = 8
        base_demand = 8000      
        elasticity = -3.5       
        marketing_power = 40    
    else: 
        base_price = 35
        base_demand = 3000
        elasticity = -4.5
        marketing_power = 20

    # Generating features
    price = np.random.uniform(base_price * 0.5, base_price * 1.5, n_rows)
    marketing = np.random.uniform(1000, 50000, n_rows)
    competitor_price = price + np.random.normal(0, base_price * 0.15, n_rows)
    
    # Calculating Demand
    price_impact = (price / base_price) ** elasticity
    marketing_impact = marketing_power * np.log1p(marketing)
    demand = (base_demand * price_impact) + marketing_impact
    noise = np.random.normal(0, base_demand * 0.1, n_rows)
    demand = np.maximum(demand + noise, 0)
    
    df = pd.DataFrame({
        'Price': price,
        'Marketing_Spend': marketing,
        'Competitor_Price': competitor_price,
        'Demand': demand
    })
    return df, base_price

# 3) Model Training
@st.cache_resource
def train_model(df):
    X = df.drop(columns=['Demand'])
    y = df['Demand']
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model, X

# Get COGS function
def get_cogs_pct(industry):
    if "Food" in industry: return 0.85 
    elif "Cosmetics" in industry: return 0.20 
    elif "Personal Care" in industry: return 0.50 
    elif "Tech" in industry: return 0.40
    else: return 0.15 # Luxury

# 4) Sidebar Controls
st.sidebar.header("🕹️ Simulation Controls")

industry_choice = st.sidebar.selectbox(
    "Select Industry / Sector", 
    ["FMCG (Food)", "FMCG (Cosmetics)", "FMCG (Personal Care)", "Tech (Gadgets)", "Fashion (Luxury)"]
)

# Loading Data
df, sim_base_price = generate_data(industry_choice)
model, X_train = train_model(df)

# Dynamic Ranges
p_min = int(sim_base_price * 0.3)
p_max = int(sim_base_price * 2.5)
p_def = int(sim_base_price)

st.sidebar.divider()

# Competitor Prices
st.sidebar.subheader("1️⃣ Current Market & Strategy")
current_price = st.sidebar.number_input("Your Current Price ($)", min_value=p_min, max_value=p_max, value=p_def)
current_marketing = st.sidebar.number_input("Your Current Marketing ($)", min_value=1000, max_value=50000, value=10000)
competitor_price = st.sidebar.number_input("Competitor Price ($)", min_value=p_min, max_value=p_max, value=p_def)

st.sidebar.divider()

st.sidebar.subheader("2️⃣ New Strategy (Simulation)")
new_price = st.sidebar.number_input("New Price ($)", min_value=p_min, max_value=p_max, value=p_def, key="new_p")
new_marketing = st.sidebar.number_input("New Marketing ($)", min_value=1000, max_value=50000, value=15000, key="new_m")

# Creating DataFrames
row_current = pd.DataFrame({'Price': [current_price], 'Marketing_Spend': [current_marketing], 'Competitor_Price': [competitor_price]})
row_new = pd.DataFrame({'Price': [new_price], 'Marketing_Spend': [new_marketing], 'Competitor_Price': [competitor_price]})

# 5) Main Dashboard
st.title("💡 Business Strategy Simulator")
st.markdown(f"Simulating: **{industry_choice}**")

# Predictions
pred_demand_curr = model.predict(row_current)[0]
pred_demand_new = model.predict(row_new)[0]

rev_curr = pred_demand_curr * current_price
rev_new = pred_demand_new * new_price

# Calculate Profit
cogs_pct = get_cogs_pct(industry_choice)
profit_curr = rev_curr - current_marketing - (pred_demand_curr * (current_price * cogs_pct))
profit_new = rev_new - new_marketing - (pred_demand_new * (new_price * cogs_pct))

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("New Demand", f"{int(pred_demand_new):,}", delta=f"{int(pred_demand_new - pred_demand_curr):,}")
with col2:
    st.metric("New Revenue", f"${rev_new:,.2f}", delta=f"${rev_new - rev_curr:,.2f}")
with col3:
    st.metric("New Profit", f"${profit_new:,.2f}", delta=f"${profit_new - profit_curr:,.2f}")

st.divider()

# 6) Explainability
st.subheader("🤖 What is driving the result?")
explainer = shap.Explainer(model)
shap_values = explainer(row_new)
features = row_new.columns
impacts = shap_values.values[0]
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in impacts]
fig_shap, ax = plt.subplots(figsize=(10, 3))
ax.barh(features, impacts, color=colors)
ax.axvline(0, color='black', linewidth=1, linestyle='--')
st.pyplot(fig_shap)

st.divider()

## 7) AI Optimization
st.subheader("🚀 AI Consultant")
st.write("Want to know the perfect price? Let the AI simulate 1,000 scenarios for you.")


def format_delta(val):
    if val >= 0:
        return f"+${val:.2f}"
    else:
        return f"-${abs(val):.2f}"

if st.button("✨ Find Optimal Strategy"):
    
    # 1. Grid Search
    opt_prices = np.linspace(p_min, p_max, 50)
    opt_marketing = np.linspace(1000, 50000, 20)
    
    combinations = list(itertools.product(opt_prices, opt_marketing))
    batch_df = pd.DataFrame(combinations, columns=['Price', 'Marketing_Spend'])
    batch_df['Competitor_Price'] = competitor_price 
    
    # 2. Predict
    batch_demand = model.predict(batch_df)
    batch_revenue = batch_demand * batch_df['Price']
    batch_cost = batch_df['Marketing_Spend'] + (batch_demand * batch_df['Price'] * cogs_pct)
    batch_profit = batch_revenue - batch_cost
    
    # 3. Find Max
    max_idx = np.argmax(batch_profit)
    best_case = batch_df.iloc[max_idx]
    best_profit = batch_profit[max_idx]
    
    # 4. Display Results 
    st.success(f"Strategy Found! To maximize profit, you should set:")
    
    o_col1, o_col2, o_col3 = st.columns(3)
    
    # Apply the helper to the 'delta' parameter
    o_col1.metric(
        "Optimal Price", 
        f"${best_case['Price']:.2f}", 
        delta=format_delta(best_case['Price'] - new_price)
    )
    
    o_col2.metric(
        "Optimal Marketing", 
        f"${best_case['Marketing_Spend']:.0f}", 
        delta=format_delta(best_case['Marketing_Spend'] - new_marketing)
    )
    
    o_col3.metric(
        "Potential Profit", 
        f"${best_profit:,.2f}", 
        delta=format_delta(best_profit - profit_new)
    )
    
    st.write(f"**Insight:** Moving to this strategy could increase your profit by **${best_profit - profit_new:,.2f}** compared to your simulation.")