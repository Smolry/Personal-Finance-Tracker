# In scripts/create_sample_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def create_sample_transactions(num_transactions=100):
    """
    Create a sample transaction dataset for testing
    """
    # Common merchants by category
    merchants = {
        'Groceries': ['Walmart', 'Safeway', 'Trader Joe\'s', 'Whole Foods', 'Kroger'],
        'Dining': ['Starbucks', 'McDonald\'s', 'Chipotle', 'Pizza Hut', 'Local Restaurant'],
        'Transportation': ['Uber', 'Lyft', 'Gas Station', 'Car Repair', 'Public Transit'],
        'Utilities': ['Electric Company', 'Water Bill', 'Gas Bill', 'Internet Provider', 'Phone Bill'],
        'Entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Amazon Prime', 'Gaming'],
        'Shopping': ['Amazon', 'Target', 'Best Buy', 'Macy\'s', 'Online Shopping'],
        'Income': ['Salary Deposit', 'Side Gig', 'Refund', 'Interest', 'Dividend']
    }
    
    # Define typical spending ranges by category
    amount_ranges = {
        'Groceries': (20, 150),
        'Dining': (10, 75),
        'Transportation': (20, 60),
        'Utilities': (40, 200),
        'Entertainment': (10, 50),
        'Shopping': (20, 200),
        'Income': (1000, 5000)
    }
    
    # Create date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    
    # Generate transactions
    transactions = []
    
    for _ in range(num_transactions):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        
        min_amount, max_amount = amount_ranges[category]
        amount = round(random.uniform(min_amount, max_amount), 2)
        
        # Income is positive, expenses are negative
        if category == 'Income':
            amount = abs(amount)
        else:
            amount = -abs(amount)
        
        date = random.choice(date_range)
        
        transactions.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': merchant,
            'amount': amount,
            'category': category
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add recurring transactions (monthly bills)
    recurring_merchants = [
        ('Netflix', 'Entertainment', -15.99),
        ('Internet Provider', 'Utilities', -89.99),
        ('Phone Bill', 'Utilities', -75.00),
        ('Rent Payment', 'Housing', -1200.00),
        ('Salary Deposit', 'Income', 3500.00)
    ]
    
    for merchant, category, amount in recurring_merchants:
        for month in range(6):
            recurring_date = end_date - timedelta(days=30 * month)
            # Adjust day to be consistent for each recurring transaction
            recurring_date = recurring_date.replace(day=min(recurring_date.day, 28))
            
            transactions.append({
                'date': recurring_date.strftime('%Y-%m-%d'),
                'description': merchant,
                'amount': amount,
                'category': category
            })
    
    # Convert to DataFrame again with the added recurring transactions
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Save the sample data
    output_path = os.path.join('data', 'raw', 'sample_transactions.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample data with {len(df)} transactions saved to {output_path}")
    
if __name__ == "__main__":
    create_sample_transactions(100)