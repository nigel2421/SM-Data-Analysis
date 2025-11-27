import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import sqlite3
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Soil Merchants ERP", layout="wide", page_icon="🏗️")

# --- DATABASE SETUP ---
if not os.path.exists('database'):
    os.makedirs('database')

DB_PATH = 'database/soil_merchants.db'

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def save_to_db(df, table_name):
    conn = get_db_connection()
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def load_from_db(table_name):
    conn = get_db_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        # Fix Dates
        for col in df.columns:
            if 'Date' in col or 'Time' in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except:
        return pd.DataFrame()

# --- HELPERS ---
def clean_currency(df, cols):
    """Removes Ksh, commas, spaces."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('Ksh', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.replace(' ', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def convert_df(df):
    """Converts a dataframe to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')

# --- SIDEBAR: INTEGRATED DATA MANAGEMENT ---
with st.sidebar:
    st.header("🗄️ Data Control Center")
    st.write("Download the template, fill it, then upload.")
    st.markdown("---")

    # --- 1. INVENTORY SECTION ---
    st.subheader("1. Inventory")
    inv_tpl = pd.DataFrame({
        'Item Name': ['Green Valley - Plot 1', 'Skyline Apts - Unit 404'],
        'Selling Rate': [450000, 8500000],
        'Stock On Hand': [1, 1]
    })
    st.download_button("📥 Get Inventory Template", data=convert_df(inv_tpl), file_name="inventory_template.csv")
    up_inv = st.file_uploader("Upload Inventory CSV", type=['csv'])
    st.markdown("---")

    # --- 2. FINANCE SECTION ---
    st.subheader("2. Revenue / Finance")
    fin_tpl = pd.DataFrame({
        'Invoice Date': ['14/07/2025', '20/07/2025'],
        'Invoice Number': ['INV-001', 'INV-002'],
        'Customer Name': ['John Doe', 'Jane Smith'],
        'Item Name': ['Green Valley - Plot 1', 'Consultation Fee'],
        'Invoice Amount': [150000, 50000],
        'Status': ['Paid', 'Overdue']
    })
    st.download_button("📥 Get Finance Template", data=convert_df(fin_tpl), file_name="finance_template.csv")
    up_fin = st.file_uploader("Upload Revenue CSV", type=['csv'])
    st.markdown("---")

    # --- 3. EXPENSES SECTION ---
    st.subheader("3. Expenses")
    exp_tpl = pd.DataFrame({
        'Date': ['01/08/2025', '05/08/2025'],
        'Category': ['Marketing', 'Construction Material'],
        'Amount': [25000, 150000],
        'Reference': ['EXP-001', 'EXP-002']
    })
    st.download_button("📥 Get Expense Template", data=convert_df(exp_tpl), file_name="expense_template.csv")
    up_exp = st.file_uploader("Upload Expense CSV", type=['csv'])
    st.markdown("---")

    # --- 4. LEADS / CRM SECTION ---
    st.subheader("4. Leads & CRM")
    lead_tpl = pd.DataFrame({
        'Lead Name': ['Michael Scott', 'Dwight Schrute'],
        'Date': ['10/08/2025', '11/08/2025'],
        'Source': ['Facebook', 'Referral'],
        'Status': ['New', 'Site Visit'],
        'Agent': ['Agent A', 'Agent B']
    })
    st.download_button("📥 Get CRM Template", data=convert_df(lead_tpl), file_name="leads_template.csv")
    up_leads = st.file_uploader("Upload Leads CSV", type=['csv'])
    st.markdown("---")

    # --- PROCESS BUTTON ---
    if st.button("💾 Process & Save All Files", type="primary"):
        # 1. INVENTORY PROCESSING
        if up_inv:
            df = pd.read_csv(up_inv)
            df.columns = df.columns.str.strip().str.title()
            df = clean_currency(df, ['Selling Rate', 'Stock On Hand'])
            # Auto-extract Project
            if 'Project' not in df.columns and 'Item Name' in df.columns:
                df['Project'] = df['Item Name'].str.split('-').str[0]
            save_to_db(df, 'inventory')
        
        # 2. FINANCE PROCESSING
        if up_fin:
            df = pd.read_csv(up_fin)
            df.columns = df.columns.str.strip().str.title()
            mapper = {'Invoice Date': 'Date', 'Invoice Amount': 'Total', 'Balance': 'Total'}
            df.rename(columns=mapper, inplace=True)
            if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = clean_currency(df, ['Total'])
            if 'Status' not in df.columns: df['Status'] = 'Paid'
            # Auto-extract Project
            if 'Project' not in df.columns and 'Item Name' in df.columns:
                df['Project'] = df['Item Name'].str.split('-').str[0]
            save_to_db(df, 'finance')

        # 3. EXPENSE PROCESSING
        if up_exp:
            df = pd.read_csv(up_exp)
            df.columns = df.columns.str.strip().str.title()
            if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = clean_currency(df, ['Amount'])
            save_to_db(df, 'expenses')

        # 4. LEADS PROCESSING
        if up_leads:
            df = pd.read_csv(up_leads)
            df.columns = df.columns.str.strip().str.title()
            if 'Status' in df.columns:
                df['Status'] = df['Status'].str.title().str.strip()
            save_to_db(df, 'leads')
            
        st.success("✅ All data successfully processed and saved!")
        st.rerun()

# --- LOAD DATA ---
df_inv = load_from_db('inventory')
df_fin = load_from_db('finance')
df_exp = load_from_db('expenses')
df_leads = load_from_db('leads')

# --- MAIN DASHBOARD ---
st.title("🏗️ Soil Merchants ERP System")

tabs = st.tabs(["💰 P&L Overview", "🏗️ Project Analysis", "💸 Collections (AR)", "👥 CRM Funnel", "🤖 AI Analyst"])

# --- TAB 1: P&L OVERVIEW ---
with tabs[0]:
    st.subheader("Global Financial Health")
    total_rev = df_fin[df_fin['Status']=='Paid']['Total'].sum() if not df_fin.empty else 0
    total_exp = df_exp['Amount'].sum() if not df_exp.empty else 0
    net_profit = total_rev - total_exp
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Revenue", f"Ksh {total_rev:,.0f}")
    k2.metric("Total Expenses", f"Ksh {total_exp:,.0f}", delta_color="inverse")
    k3.metric("Net Profit", f"Ksh {net_profit:,.0f}")

    if not df_fin.empty:
        rev_trend = df_fin.groupby(pd.Grouper(key='Date', freq='M'))['Total'].sum().reset_index()
        rev_trend['Type'] = 'Income'
        if not df_exp.empty:
            exp_trend = df_exp.groupby(pd.Grouper(key='Date', freq='M'))['Amount'].sum().reset_index()
            exp_trend.rename(columns={'Amount': 'Total'}, inplace=True)
            exp_trend['Type'] = 'Expense'
            rev_trend = pd.concat([rev_trend, exp_trend])
        st.plotly_chart(px.bar(rev_trend, x='Date', y='Total', color='Type', barmode='group', title="Monthly Cash Flow"), use_container_width=True)

# --- TAB 2: PROJECT ANALYSIS ---
with tabs[1]:
    st.subheader("Project-Level Profitability")
    if not df_fin.empty and 'Project' in df_fin.columns:
        proj_rev = df_fin[df_fin['Status']=='Paid'].groupby('Project')['Total'].sum().reset_index()
        st.plotly_chart(px.bar(proj_rev, x='Project', y='Total', color='Total', title="Revenue by Project"), use_container_width=True)
    else:
        st.info("Upload Inventory/Finance data with 'Item Name' formatted like 'ProjectName - Item'")

# --- TAB 3: COLLECTIONS (AR) ---
with tabs[2]:
    st.subheader("⚠️ Debt & Collections")
    if not df_fin.empty and 'Status' in df_fin.columns:
        unpaid = df_fin[df_fin['Status'].isin(['Overdue', 'Unpaid', 'Pending', 'Late'])].copy()
        if not unpaid.empty:
            total_debt = unpaid['Total'].sum()
            st.metric("Outstanding Debt", f"Ksh {total_debt:,.0f}", delta="Action Required", delta_color="inverse")
            
            now = pd.Timestamp.now()
            unpaid['Days_Overdue'] = (now - unpaid['Date']).dt.days
            
            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(unpaid[['Customer Name', 'Total', 'Days_Overdue']].sort_values('Days_Overdue', ascending=False))
            with c2:
                # Top Debtors Chart
                top_debt = unpaid.groupby('Customer Name')['Total'].sum().nlargest(10).reset_index()
                st.plotly_chart(px.bar(top_debt, x='Total', y='Customer Name', orientation='h', title="Top 10 Debtors"), use_container_width=True)
        else:
            st.success("No overdue payments!")

# --- TAB 4: CRM FUNNEL ---
with tabs[3]:
    st.subheader("Lead Conversions")
    if not df_leads.empty:
        stages = ['New', 'Contacted', 'Site Visit', 'Negotiation', 'Converted']
        funnel_data = []
        for s in stages:
            count = len(df_leads[df_leads['Status'].str.contains(s, case=False, na=False)])
            funnel_data.append(count)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(go.Figure(go.Funnel(y=stages, x=funnel_data)), use_container_width=True)
        with c2:
            st.markdown("**Source Efficiency**")
            source_stats = df_leads.groupby('Source')['Status'].apply(lambda x: (x.str.contains('Converted').sum() / len(x))*100).reset_index(name='Conv_Rate')
            st.plotly_chart(px.bar(source_stats, x='Source', y='Conv_Rate'), use_container_width=True)

# --- TAB 5: AI ANALYST ---
with tabs[4]:
    st.header("🤖 Local AI Assistant")
    user_query = st.text_input("Ask about: Debt, Top Project, Forecasts")
    
    if user_query:
        q = user_query.lower()
        if 'debt' in q:
            val = df_fin[df_fin['Status']=='Overdue']['Total'].sum() if not df_fin.empty else 0
            st.error(f"Total Overdue Debt: Ksh {val:,.0f}")
        elif 'forecast' in q:
            if len(df_fin) > 5:
                temp = df_fin.groupby(pd.Grouper(key='Date', freq='M'))['Total'].sum().reset_index()
                model = LinearRegression()
                model.fit(temp.index.values.reshape(-1,1), temp['Total'])
                pred = model.predict([[len(temp)+1]])[0]
                st.info(f"Next Month Forecast: Ksh {pred:,.0f}")
            else:
                st.warning("Not enough data to forecast.")
        else:
            st.write("Try asking: 'How much debt?' or 'Revenue forecast'")