import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import etl # Keep this for the fallback mock data

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Property Strategist", layout="wide", page_icon="🏢")

st.title("🏢 Real Estate AI Command Center")
st.markdown("Recommending strategies based on *Inventory Velocity* and *Financial Trends*.")

# --- SIDEBAR: DATA LOADER ---
with st.sidebar:
    st.header("🎛️ Control Center")
    st.write("Upload your actual data below.")
    
    uploaded_inventory = st.file_uploader("Upload Inventory (CSV)", type=['csv'])
    uploaded_finance = st.file_uploader("Upload Transactions (CSV)", type=['csv'])
    
    data_source = "Mock"
    
    if uploaded_inventory and uploaded_finance:
        try:
            df_props = pd.read_csv(uploaded_inventory)
            df_pay = pd.read_csv(uploaded_finance)
            # Basic cleaning to ensure dates are dates
            df_pay['Transaction_Date'] = pd.to_datetime(df_pay['Transaction_Date'])
            data_source = "Real"
            st.success("✅ Real Data Loaded")
        except Exception as e:
            st.error(f"Error reading files: {e}")
            st.stop()
    else:
        st.info("Using Simulation Data (Upload files to override)")
        # Fallback to the mock data generator we built earlier
        df_props, df_pay = etl.generate_real_estate_data()
        # Rename columns to match the 'Real' schema for consistency
        df_pay.rename(columns={'date': 'Transaction_Date'}, inplace=True)
        data_source = "Mock"

# --- AI ENGINE: PREDICTION FUNCTIONS ---
def run_revenue_prediction(payment_df):
    """
    Uses Linear Regression to predict the next 3 months of revenue
    based on historical transaction data.
    """
    # 1. Aggregate revenue by month
    monthly_rev = payment_df[payment_df['status']=='Paid'].set_index('Transaction_Date').resample('M')['amount'].sum().reset_index()
    monthly_rev['Month_Index'] = range(len(monthly_rev))
    
    if len(monthly_rev) < 2:
        return None, None # Not enough data to predict

    # 2. Train the Model (The AI Part)
    X = monthly_rev[['Month_Index']] # Input: Time
    y = monthly_rev['amount']        # Output: Money
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Predict Future (Next 3 months)
    last_index = monthly_rev['Month_Index'].max()
    future_indexes = np.array([[last_index + 1], [last_index + 2], [last_index + 3]])
    predictions = model.predict(future_indexes)
    
    return monthly_rev, predictions

# --- DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "🤖 AI Forecast & Insights", "📂 Data Inspector"])

with tab1:
    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)
    
    total_rev = df_pay[df_pay['status']=='Paid']['amount'].sum()
    total_sales = len(df_props[(df_props['category']=='Sale') & (df_props['status']=='Sold')])
    occupancy = len(df_props[(df_props['category']=='Rent') & (df_props['status']=='Occupied')])
    total_rentals = len(df_props[df_props['category']=='Rent'])
    occ_rate = (occupancy/total_rentals * 100) if total_rentals > 0 else 0
    
    col1.metric("Total Revenue Collected", f"${total_rev:,.0f}")
    col2.metric("Sales Closed", total_sales)
    col3.metric("Rental Occupancy", f"{occ_rate:.1f}%")
    col4.metric("Data Source", data_source)

    # REVENUE CHART
    st.subheader("Cash Flow Trend")
    rev_trend = df_pay.groupby(pd.Grouper(key='Transaction_Date', freq='M'))['amount'].sum().reset_index()
    fig_trend = px.line(rev_trend, x='Transaction_Date', y='amount', title="Monthly Revenue Inflow", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("🤖 AI Analysis & Feedback")
    
    col_ai_left, col_ai_right = st.columns([2, 1])
    
    with col_ai_left:
        st.subheader("Revenue Forecasting (Linear Regression)")
        hist_data, predictions = run_revenue_prediction(df_pay)
        
        if hist_data is not None:
            # Create a dataframe for the future
            last_date = hist_data['Transaction_Date'].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
            
            df_future = pd.DataFrame({
                'Transaction_Date': future_dates,
                'amount': predictions,
                'Type': 'AI Prediction'
            })
            hist_data['Type'] = 'Actual'
            
            # Combine for charting
            df_combined = pd.concat([hist_data[['Transaction_Date', 'amount', 'Type']], df_future])
            
            fig_pred = px.line(df_combined, x='Transaction_Date', y='amount', color='Type', 
                               line_dash='Type', markers=True, title="Projected vs Actual Revenue")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            projected_total = sum(predictions)
            st.info(f"💡 **AI Projection:** Based on your current trajectory, the model predicts **${projected_total:,.0f}** in revenue over the next 90 days.")
        else:
            st.warning("Not enough historical data to generate predictions.")

    with col_ai_right:
        st.subheader("💡 Strategic Recommendations")
        
        # LOGIC-BASED "AI" FEEDBACK
        recommendations = []
        
        # 1. Check Sales Velocity
        unsold_land = len(df_props[(df_props['project'].str.contains('Land')) & (df_props['status']=='Available')])
        if unsold_land > 10:
            recommendations.append(f"⚠️ **High Land Inventory:** You have {unsold_land} plots unsold. Consider a 'Flash Sale' marketing campaign to unlock capital.")
            
        # 2. Check Rental Arrears
        late_payments = len(df_pay[(df_pay['type']=='Rent') & (df_pay['status']=='Late')])
        if late_payments > 5:
            recommendations.append(f"⚠️ **Cash Flow Risk:** {late_payments} rental payments were late recently. Direct Property Management to enforce strict collection policies.")
            
        # 3. Check Occupancy
        if occ_rate < 85:
             recommendations.append(f"📉 **Low Occupancy ({occ_rate:.1f}%):** Your rentals are underperforming. Review pricing strategy or renovation needs.")
        else:
             recommendations.append(f"✅ **Strong Occupancy ({occ_rate:.1f}%):** You have pricing power. Consider increasing rents by 3-5% on lease renewals.")

        # Display recommendations
        for rec in recommendations:
            st.write(rec)
            st.markdown("---")

with tab3:
    st.write("Check your raw data here to ensure accuracy.")
    st.write("### Property Inventory")
    st.dataframe(df_props)
    st.write("### Financial Transactions")
    st.dataframe(df_pay)