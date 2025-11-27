import pandas as pd
from sqlalchemy import create_engine
import random
from datetime import datetime, timedelta
import numpy as np

# DATABASE CONNECTION
db_engine = create_engine('sqlite:///database/local_data.db')

def generate_real_estate_data():
    print("🏗️  Generating Property Data...")

    # --- CONFIGURATION ---
    projects = ['Skyline Heights (Apts)', 'Green Valley (Land)', 'Urban Oasis (Rentals)', 'Managed Estates (3rd Party)']
    unit_types = {
        'Skyline Heights (Apts)': {'type': 'Sale', 'price_range': (80000, 150000)},
        'Green Valley (Land)': {'type': 'Sale', 'price_range': (20000, 40000)},
        'Urban Oasis (Rentals)': {'type': 'Rent', 'price_range': (500, 1200)}, # Monthly Rent
        'Managed Estates (3rd Party)': {'type': 'Rent', 'price_range': (800, 2000)} # Monthly Rent
    }

    properties = []
    payments = []
    
    # Generate 100 Units
    for i in range(1, 101):
        project = random.choice(projects)
        config = unit_types[project]
        
        unit_id = f"{project[:3].upper()}-{100+i}"
        price = random.randint(config['price_range'][0], config['price_range'][1])
        
        # Determine Status
        if config['type'] == 'Sale':
            status = np.random.choice(['Available', 'Sold', 'Reserved'], p=[0.3, 0.6, 0.1])
            is_managed = False
        else:
            status = np.random.choice(['Vacant', 'Occupied'], p=[0.1, 0.9])
            is_managed = True if project == 'Managed Estates (3rd Party)' else False

        # Create Property Record
        prop_row = {
            'unit_id': unit_id,
            'project': project,
            'category': config['type'],
            'list_price': price, # Selling price OR Monthly Rent
            'status': status,
            'is_managed': is_managed, # True if we only get 5% fee
            'size_sqft': random.randint(500, 2000)
        }
        properties.append(prop_row)

        # GENERATE PAYMENTS (Financial History)
        # If Sold, generate installments. If Occupied, generate rent.
        if status == 'Sold':
            # Simulate 3 installments
            for m in range(1, 4):
                pay_date = datetime.now() - timedelta(days=m*30)
                payments.append({
                    'unit_id': unit_id,
                    'date': pay_date,
                    'amount': price * 0.2, # 20% installment
                    'type': 'Installment',
                    'status': 'Paid'
                })
        
        elif status == 'Occupied':
            # Simulate last 6 months rent
            for m in range(0, 6):
                pay_date = datetime.now() - timedelta(days=m*30)
                # Introduce "Late" or "Arrears" logic for Data Science magic
                pay_status = np.random.choice(['Paid', 'Late', 'Unpaid'], p=[0.8, 0.15, 0.05])
                
                payments.append({
                    'unit_id': unit_id,
                    'date': pay_date,
                    'amount': price,
                    'type': 'Rent',
                    'status': pay_status
                })

    # Convert to DataFrames
    df_props = pd.DataFrame(properties)
    df_pay = pd.DataFrame(payments)
    
    # Ensure dates are datetime
    df_pay['date'] = pd.to_datetime(df_pay['date'])

    return df_props, df_pay

def run_pipeline():
    # 1. Generate
    df_props, df_pay = generate_real_estate_data()
    
    # 2. Load to Database (Two separate tables)
    df_props.to_sql('properties', db_engine, if_exists='replace', index=False)
    df_pay.to_sql('payments', db_engine, if_exists='replace', index=False)
    
    print("✅ SUCCESS: Real Estate Data & Financials updated.")

if __name__ == "__main__":
    run_pipeline()