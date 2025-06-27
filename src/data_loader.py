import pandas as pd
import os
from datetime import datetime

def get_available_datasets():
    """
    Get a list of available CSV files in the data/raw directory
    """
    data_path = os.path.join('data', 'raw', '*.csv')
    files = glob.glob(data_path)
    return [os.path.basename(f) for f in files]

def load_csv_transactions(file_path=None, filename=None):
    """
    Load transactions from CSV file.
    
    Args:
        file_path: Direct path to CSV file (for uploaded files)
        filename: Name of file in data/raw directory
    """
    if filename and not file_path:
        file_path = os.path.join('data', 'raw', filename)
    
    df = pd.read_csv(file_path)
    
    # Rename columns to standardized format
    # This example assumes CSV has columns: Date, Description, Amount
    # Adjust based on your actual CSV format
    column_mapping = {
        'Date': 'date',
        'Description': 'description',
        'Amount': 'amount'
    }
    
# Check which columns exist in the dataset
    valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    
    df = df.rename(columns=valid_columns)
    
    # If the required columns don't exist with those exact names,
    # try to find similar columns based on common naming patterns
    if 'date' not in df.columns:
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: 'date'})
    
    if 'description' not in df.columns:
        desc_candidates = [col for col in df.columns if 'desc' in col.lower() or 'name' in col.lower() or 'merch' in col.lower()]
        if desc_candidates:
            df = df.rename(columns={desc_candidates[0]: 'description'})
    
    if 'amount' not in df.columns:
        amount_candidates = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower() or 'price' in col.lower()]
        if amount_candidates:
            df = df.rename(columns={amount_candidates[0]: 'amount'})
    
    # Make sure required columns exist
    required_columns = ['date', 'description', 'amount']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file")
    
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Handle missing dates
    df = df.dropna(subset=['date'])
    
    # Ensure amount is a float and uses consistent sign convention
    df['amount'] = df['amount'].astype(float)
    
    # Add a placeholder category column for later classification if it doesn't exist
    if 'category' not in df.columns:
        df['category'] = 'Uncategorized'
    
    return df

def save_transactions_to_sqlite(df, db_path='data/finance.db'):
    """
    Save transactions to SQLite database
    """
    import sqlite3
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    
    # Create transactions table if it doesn't exist
    conn.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY,
        date TEXT,
        description TEXT,
        amount REAL,
        category TEXT
    )
    ''')
    
    # Save transactions to database
    df.to_sql('transactions', conn, if_exists='append', index=False)
    
    conn.close()
    
    return True