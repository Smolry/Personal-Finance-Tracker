"""
Generate a large, realistic transaction dataset for training ML models
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_dataset(num_transactions=1500, months=12):
    """
    Generate a large, realistic transaction dataset
    
    Args:
        num_transactions: Total number of random transactions
        months: Number of months of history
    """
    
    # Expanded merchant database with more realistic names
    merchants = {
        'Groceries': [
            'Walmart Supercenter', 'Target', 'Whole Foods Market', 'Trader Joes',
            'Safeway', 'Kroger', 'Albertsons', 'Stop & Shop', 'Food Lion',
            'Publix', 'HEB', 'Wegmans', 'Aldi', 'Costco', 'Sams Club'
        ],
        'Dining': [
            'Starbucks', 'McDonalds', 'Chipotle Mexican Grill', 'Subway',
            'Panera Bread', 'Chick-fil-A', 'Taco Bell', 'Pizza Hut',
            'Dominos Pizza', 'The Cheesecake Factory', 'Olive Garden',
            'Red Lobster', 'Buffalo Wild Wings', 'DoorDash', 'Uber Eats',
            'Grubhub', 'Local Cafe', 'Thai Restaurant', 'Sushi Bar'
        ],
        'Transportation': [
            'Uber', 'Lyft', 'Shell Gas Station', 'Chevron', 'BP Gas',
            'Exxon Mobil', 'Public Transit', 'Metro Card', 'Parking Garage',
            'Airport Parking', 'Car Wash', 'Auto Repair Shop', 'Jiffy Lube'
        ],
        'Utilities': [
            'Electric Company', 'City Water Department', 'Natural Gas Co',
            'Comcast Internet', 'AT&T', 'Verizon Wireless', 'T-Mobile',
            'Spectrum Internet', 'Trash Collection Service'
        ],
        'Entertainment': [
            'Netflix', 'Spotify Premium', 'Disney Plus', 'HBO Max',
            'Amazon Prime Video', 'Hulu', 'Apple Music', 'YouTube Premium',
            'AMC Theaters', 'Regal Cinemas', 'PlayStation Store',
            'Xbox Live', 'Steam', 'Nintendo eShop', 'Gym Membership'
        ],
        'Shopping': [
            'Amazon.com', 'eBay', 'Best Buy', 'Macys', 'Nordstrom',
            'Home Depot', 'Lowes', 'IKEA', 'Target', 'Old Navy',
            'Gap', 'H&M', 'Zara', 'Nike', 'Adidas', 'Apple Store',
            'Microsoft Store', 'Staples', 'Office Depot'
        ],
        'Health': [
            'CVS Pharmacy', 'Walgreens', 'Rite Aid', 'Medical Clinic',
            'Dentist Office', 'Eye Doctor', 'Physical Therapy',
            'Health Insurance Premium', 'Prescription Refill',
            'Lab Tests', 'Fitness Center', 'Yoga Studio'
        ],
        'Housing': [
            'Rent Payment', 'Mortgage Payment', 'HOA Fees',
            'Property Tax', 'Home Insurance', 'Apartment Maintenance'
        ],
        'Travel': [
            'Delta Airlines', 'United Airlines', 'American Airlines',
            'Southwest Airlines', 'Marriott Hotel', 'Hilton Hotels',
            'Airbnb', 'Expedia', 'Hertz Rental Car', 'Enterprise Rent-A-Car'
        ],
        'Education': [
            'University Tuition', 'Textbook Store', 'Online Course',
            'Udemy', 'Coursera', 'School Supplies', 'Student Loan Payment'
        ],
        'Subscriptions': [
            'Adobe Creative Cloud', 'Microsoft 365', 'New York Times',
            'Wall Street Journal', 'Audible', 'Kindle Unlimited',
            'LinkedIn Premium', 'Dropbox Pro', 'Google One Storage'
        ],
        'Income': [
            'Payroll Deposit - Company Inc', 'Freelance Payment',
            'Bank Interest', 'Dividend Payment', 'Tax Refund',
            'Cashback Reward', 'Gift Money', 'Side Hustle Income'
        ]
    }
    
    # Realistic amount ranges by category
    amount_ranges = {
        'Groceries': (15, 180),
        'Dining': (8, 95),
        'Transportation': (5, 75),
        'Utilities': (35, 250),
        'Entertainment': (8, 60),
        'Shopping': (15, 350),
        'Health': (20, 200),
        'Housing': (800, 2500),
        'Travel': (100, 1500),
        'Education': (50, 5000),
        'Subscriptions': (5, 80),
        'Income': (500, 6000)
    }
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    transactions = []
    
    # Generate random transactions
    for _ in range(num_transactions):
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        
        min_amt, max_amt = amount_ranges[category]
        amount = round(random.uniform(min_amt, max_amt), 2)
        
        # Income is positive, expenses are negative
        if category == 'Income':
            amount = abs(amount)
        else:
            amount = -abs(amount)
        
        # Random date within range
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        date = start_date + timedelta(days=random_days)
        
        transactions.append({
            'date': date.strftime('%Y-%m-%d'),
            'description': merchant,
            'amount': amount,
            'category': category
        })
    
    # Add realistic recurring transactions
    recurring_items = [
        ('Netflix', 'Entertainment', -15.99, 1),  # day of month
        ('Spotify Premium', 'Entertainment', -9.99, 1),
        ('Rent Payment', 'Housing', -1200.00, 1),
        ('Electric Company', 'Utilities', -120.00, 5),
        ('Comcast Internet', 'Utilities', -89.99, 10),
        ('Verizon Wireless', 'Utilities', -75.00, 15),
        ('Gym Membership', 'Entertainment', -45.00, 1),
        ('Amazon Prime', 'Subscriptions', -14.99, 12),
        ('Adobe Creative Cloud', 'Subscriptions', -54.99, 20),
        ('Payroll Deposit - Company Inc', 'Income', 3500.00, 1),
        ('Payroll Deposit - Company Inc', 'Income', 3500.00, 15),
    ]
    
    for merchant, category, amount, day_of_month in recurring_items:
        current_date = start_date
        while current_date <= end_date:
            # Set to specific day of month
            try:
                recurring_date = current_date.replace(day=day_of_month)
            except ValueError:
                # Handle months with fewer days
                recurring_date = current_date.replace(day=min(day_of_month, 28))
            
            if start_date <= recurring_date <= end_date:
                transactions.append({
                    'date': recurring_date.strftime('%Y-%m-%d'),
                    'description': merchant,
                    'amount': amount,
                    'category': category
                })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Generated {len(df)} transactions over {months} months")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_large_dataset(num_transactions=1500, months=12)
    
    # Save to CSV
    output_path = os.path.join('data','raw','large_transactions.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
