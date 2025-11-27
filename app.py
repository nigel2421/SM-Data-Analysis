import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soil Merchants Analytics", layout="wide", page_icon="🏗️")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #2c3e50; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏗️ Soil Merchants: Strategic Intelligence")

# --- SESSION STATE ---
if 'inventory_db' not in st.session_state:
    st.session_state.inventory_db = pd.DataFrame()
if 'finance_db' not in st.session_state:
    st.session_state.finance_db = pd.DataFrame()

# --- HELPER: CLEAN CURRENCY (Fixes the "Unknown format code" error) ---
def clean_currency_column(df, col_name):
    """Removes 'Ksh', commas, and spaces, then converts to float."""
    if col_name in df.columns:
        # Force to string first, then clean
        df[col_name] = df[col_name].astype(str).str.replace('Ksh', '', regex=False)
        df[col_name] = df[col_name].str.replace(',', '', regex=False)
        df[col_name] = df[col_name].str.replace(' ', '', regex=False)
        # Convert to numeric, turning errors (like empty text) into 0
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Data Control Center")
    
    # RESET BUTTON (Critical for fixing stuck session state)
    if st.button("⚠️ Reset / Clear All Data"):
        st.session_state.inventory_db = pd.DataFrame()
        st.session_state.finance_db = pd.DataFrame()
        st.rerun()

    # UPLOADERS
    st.subheader("Upload Zoho Reports")
    uploaded_inv = st.file_uploader("Inventory (CSV)", type=['csv'])
    uploaded_fin = st.file_uploader("Finance (CSV)", type=['csv'])
    
    if st.button("🚀 Process & Merge Data"):
        
        # --- 1. PROCESS INVENTORY ---
        if uploaded_inv:
            try:
                new_inv = pd.read_csv(uploaded_inv)
                new_inv.columns = new_inv.columns.str.strip().str.title() # Clean headers
                
                # Standardize Column Names
                inv_map = {
                    'Item Name': 'Item Name',
                    'Sku': 'SKU', 'Stock On Hand': 'Stock On Hand',
                    'Selling Rate': 'Selling Rate', 'Rate': 'Selling Rate', 'Price': 'Selling Rate'
                }
                new_inv.rename(columns=inv_map, inplace=True)
                
                # FIX: Clean Numbers (Remove Ksh/Commas)
                new_inv = clean_currency_column(new_inv, 'Selling Rate')
                new_inv = clean_currency_column(new_inv, 'Stock On Hand')

                # Merge Logic
                if not st.session_state.inventory_db.empty and 'SKU' in new_inv.columns:
                    existing = st.session_state.inventory_db
                    if 'SKU' in existing.columns:
                        existing = existing[~existing['SKU'].isin(new_inv['SKU'])]
                    st.session_state.inventory_db = pd.concat([existing, new_inv], ignore_index=True)
                else:
                    st.session_state.inventory_db = new_inv
                st.success(f"Inventory: {len(new_inv)} records loaded.")
            except Exception as e:
                st.error(f"Inventory Error: {e}")

        # --- 2. PROCESS FINANCE ---
        if uploaded_fin:
            try:
                new_fin = pd.read_csv(uploaded_fin)
                new_fin.columns = new_fin.columns.str.strip().str.title()
                
                # Standardize Column Names (Fixes KeyError: Status)
                fin_map = {
                    'Invoice Status': 'Status', 'Payment Status': 'Status',
                    'Invoice Amount': 'Total', 'Balance': 'Total', 'Amount': 'Total',
                    'Invoice Date': 'Invoice Date', 'Date': 'Invoice Date'
                }
                new_fin.rename(columns=fin_map, inplace=True)

                # EMERGENCY FIX: If 'Status' still missing, create it
                if 'Status' not in new_fin.columns:
                    new_fin['Status'] = 'Paid' # Default value
                
                # FIX: Clean Numbers (Remove Ksh/Commas)
                new_fin = clean_currency_column(new_fin, 'Total')

                # FIX: Dates
                if 'Invoice Date' in new_fin.columns:
                    new_fin['Invoice Date'] = pd.to_datetime(new_fin['Invoice Date'], dayfirst=True, errors='coerce')
                    new_fin = new_fin.dropna(subset=['Invoice Date'])
                
                # Merge Logic
                if not st.session_state.finance_db.empty and 'Invoice Number' in new_fin.columns:
                    existing = st.session_state.finance_db[~st.session_state.finance_db['Invoice Number'].isin(new_fin['Invoice Number'])]
                    st.session_state.finance_db = pd.concat([existing, new_fin], ignore_index=True)
                else:
                    st.session_state.finance_db = new_fin
                st.success(f"Finance: {len(new_fin)} records loaded.")
            except Exception as e:
                st.error(f"Finance Error: {e}")

# --- DASHBOARD ---
if st.session_state.inventory_db.empty or st.session_state.finance_db.empty:
    st.info("👋 Waiting for data. Please upload files in the sidebar.")
    st.stop()

df_inv = st.session_state.inventory_db
df_fin = st.session_state.finance_db

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🤖 AI Forecast", "📂 Data"])

with tab1:
    st.subheader("Financial Snapshot")
    
    # Safe Calculations
    total_revenue = 0
    pending_revenue = 0
    
    if 'Status' in df_fin.columns and 'Total' in df_fin.columns:
        # Ensure 'Total' is numeric before summing
        total_revenue = df_fin[df_fin['Status'] == 'Paid']['Total'].sum()
        pending_revenue = df_fin[df_fin['Status'] != 'Paid']['Total'].sum()
    
    avail_stock = 0
    if 'Stock On Hand' in df_inv.columns:
        avail_stock = df_inv['Stock On Hand'].sum()
    
    total_val = 0
    if 'Selling Rate' in df_inv.columns:
        total_val = df_inv['Selling Rate'].sum()

    # Display Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Collected Revenue", f"Ksh {total_revenue:,.0f}")
    c2.metric("Pending/Overdue", f"Ksh {pending_revenue:,.0f}")
    c3.metric("Stock Available", f"{avail_stock:,.0f}")
    c4.metric("Portfolio Value", f"Ksh {total_val:,.0f}")

    # Charts
    c_left, c_right = st.columns(2)
    with c_left:
        if 'Invoice Date' in df_fin.columns and 'Total' in df_fin.columns:
            st.subheader("Cash Flow")
            trend = df_fin.groupby(pd.Grouper(key='Invoice Date', freq='M'))['Total'].sum().reset_index()
            st.plotly_chart(px.bar(trend, x='Invoice Date', y='Total'), use_container_width=True)
            
    with c_right:
        if 'Customer Name' in df_fin.columns:
            st.subheader("Top Customers")
            top = df_fin.groupby('Customer Name')['Total'].sum().nlargest(10).reset_index()
            st.plotly_chart(px.pie(top, values='Total', names='Customer Name', hole=0.4), use_container_width=True)

with tab2:
    st.header("🤖 Revenue Prediction")
    if len(df_fin) > 5 and 'Invoice Date' in df_fin.columns:
        m_rev = df_fin[df_fin['Status']=='Paid'].set_index('Invoice Date').resample('M')['Total'].sum().reset_index()
        m_rev['Idx'] = range(len(m_rev))
        
        if len(m_rev) > 2:
            model = LinearRegression()
            model.fit(m_rev[['Idx']], m_rev['Total'])
            last_idx = m_rev['Idx'].max()
            future_idx = np.array([[last_idx+1], [last_idx+2], [last_idx+3]])
            preds = model.predict(future_idx)
            
            last_date = m_rev['Invoice Date'].max()
            f_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 4)]
            df_fut = pd.DataFrame({'Invoice Date': f_dates, 'Total': preds, 'Type': 'Forecast'})
            m_rev['Type'] = 'Actual'
            
            st.plotly_chart(px.line(pd.concat([m_rev, df_fut]), x='Invoice Date', y='Total', color='Type'), use_container_width=True)
            st.success(f"Predicted Next Month: Ksh {preds[0]:,.0f}")

with tab3:
    st.dataframe(df_inv)
    st.dataframe(df_fin)