import re
import pandas as pd
from .ml_categorizer import MLCategorizer

# Define category rules based on keywords
CATEGORY_RULES = {
    'Groceries': ['supermarket', 'grocery', 'market', 'food', 'walmart', 'target'],
    'Dining': ['restaurant', 'cafe', 'coffee', 'doordash', 'ubereats', 'grubhub'],
    'Transportation': ['uber', 'lyft', 'taxi', 'transport', 'gas', 'fuel', 'parking'],
    'Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'bill'],
    'Entertainment': ['movie', 'theatre', 'netflix', 'spotify', 'amazon prime'],
    'Shopping': ['amazon', 'ebay', 'store', 'shop'],
    'Health': ['pharmacy', 'doctor', 'medical', 'healthcare', 'fitness'],
    'Income': ['payroll', 'deposit', 'salary', 'interest'],
    'Housing': ['rent', 'mortgage', 'home', 'apartment', 'lease'],
    'Travel': ['hotel', 'airbnb', 'flight', 'airline', 'travel'],
    'Education': ['tuition', 'school', 'university', 'college', 'books', 'course'],
    'Subscriptions': ['subscription', 'membership']
}

def categorize_transaction(description, ml_categorizer=None):
    """
    Categorize a transaction based on its description,
    using ML if available, otherwise falling back to rules
    """
    # Try ML categorization first if available
    if ml_categorizer and ml_categorizer.pipeline is not None:
        ml_category = ml_categorizer.predict(description)[0]
        if ml_category != "Uncategorized":
            return ml_category
    
    # Fall back to rule-based categorization
    description = description.lower()
    
    for category, keywords in CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword.lower() in description:
                return category
    
    return 'Uncategorized'

def categorize_transactions(df, use_ml=True):
    """
    Categorize all transactions in a dataframe
    """
    ml_categorizer = None
    if use_ml:
        ml_categorizer = MLCategorizer()
        ml_categorizer.load_model()
    
    # Apply categorization to each transaction
    df['category'] = df['description'].apply(
        lambda x: categorize_transaction(x, ml_categorizer)
    )
    
    # Income is typically positive amounts
    df.loc[df['amount'] > 0, 'category'] = 'Income'
    
    return df

def train_categorizer(df):
    """
    Train the ML categorizer with the given transactions
    """
    ml_categorizer = MLCategorizer()
    success = ml_categorizer.train(df)
    return success