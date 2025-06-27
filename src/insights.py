import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_spending_trends(df):
    """
    Analyze spending trends over time
    """
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for expenses only
    expenses = df[df['amount'] < 0].copy()
    expenses['amount'] = expenses['amount'].abs()  # Make positive for easier analysis
    
    # Add month and year columns
    expenses['month'] = expenses['date'].dt.month
    expenses['year'] = expenses['date'].dt.year
    expenses['month_year'] = expenses['date'].dt.strftime('%Y-%m')
    
    # Monthly spending
    monthly_spending = expenses.groupby('month_year')['amount'].sum().reset_index()
    
    # Category spending
    category_spending = expenses.groupby('category')['amount'].sum().reset_index()
    category_spending = category_spending.sort_values('amount', ascending=False)
    
    # Top merchants
    top_merchants = expenses.groupby('description')['amount'].sum().reset_index()
    top_merchants = top_merchants.sort_values('amount', ascending=False).head(10)
    
    return {
        'monthly_spending': monthly_spending,
        'category_spending': category_spending,
        'top_merchants': top_merchants
    }

def detect_recurring_expenses(df, min_occurrences=2, max_day_variance=5):
    """
    Detect recurring expenses (subscriptions, bills, etc.)
    """
    # Filter for expenses only
    expenses = df[df['amount'] < 0].copy()
    
    # Group by description and count occurrences
    merchant_counts = expenses.groupby('description').size().reset_index(name='count')
    recurring_candidates = merchant_counts[merchant_counts['count'] >= min_occurrences]
    
    recurring_expenses = []
    
    for _, row in recurring_candidates.iterrows():
        merchant = row['description']
        merchant_transactions = expenses[expenses['description'] == merchant].copy()
        
        # Sort by date
        merchant_transactions = merchant_transactions.sort_values('date')
        
        # Check if transactions occur on similar days of the month
        days_of_month = merchant_transactions['date'].dt.day
        day_variance = days_of_month.max() - days_of_month.min()
        
        if day_variance <= max_day_variance:
            avg_amount = merchant_transactions['amount'].mean()
            last_date = merchant_transactions['date'].max()
            first_date = merchant_transactions['date'].min()
            frequency = (last_date - first_date) / (len(merchant_transactions) - 1) if len(merchant_transactions) > 1 else timedelta(days=30)
            
            recurring_expenses.append({
                'merchant': merchant,
                'avg_amount': abs(avg_amount),
                'frequency_days': frequency.days,
                'last_date': last_date,
                'next_expected': last_date + frequency,
                'transactions': len(merchant_transactions)
            })
    
    return pd.DataFrame(recurring_expenses)

def suggest_budget(df):
    """
    Suggest a budget based on historical spending
    """
    # Filter for expenses only
    expenses = df[df['amount'] < 0].copy()
    expenses['amount'] = expenses['amount'].abs()
    
    # Calculate average monthly spending per category
    expenses['month_year'] = expenses['date'].dt.strftime('%Y-%m')
    monthly_category = expenses.groupby(['month_year', 'category'])['amount'].sum().reset_index()
    avg_category = monthly_category.groupby('category')['amount'].mean().reset_index()
    avg_category = avg_category.sort_values('amount', ascending=False)
    
    # Add a buffer for budget (10% more than average)
    avg_category['budget'] = avg_category['amount'] * 1.1
    
    return avg_category