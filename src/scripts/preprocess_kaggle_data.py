import pandas as pd
import os

def preprocess_personal_finance_data():
    """
    Preprocess the Personal Financial Data from Kaggle
    """
    input_path = os.path.join('data', 'raw', 'personal_transactions.csv')
    output_path = os.path.join('data', 'processed', 'preprocessed_finance_data.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return
    
    # Read the data
    df = pd.read_csv(input_path)
    
    # Rename columns to match our expected format
    column_mapping = {
        # Update these to match your actual Kaggle dataset column names
        'Date': 'date',
        'Description': 'description',
        'Amount': 'amount',
        'Category': 'category'
    }
    
    # Check which columns exist in the dataset
    valid_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    
    df = df.rename(columns=valid_columns)
    
    # Ensure required columns exist
    required_columns = ['date', 'description', 'amount']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Save the preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessed data saved to {output_path}")
    
if __name__ == "__main__":
    preprocess_personal_finance_data()