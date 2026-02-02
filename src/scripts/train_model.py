"""
Train the ML categorizer with the large dataset
"""
import pandas as pd
from src.ml_categorizer import MLCategorizer

def main():
    # Load the large dataset
    print("Loading dataset...")
    df = pd.read_csv('data/raw/large_transactions.csv')
    
    print(f"Loaded {len(df)} transactions")
    print(f"Categories: {df['category'].unique()}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    
    # Initialize categorizer
    categorizer = MLCategorizer(model_dir='models')
    
    # Train without hyperparameter tuning first (faster)
    print("\n" + "="*60)
    print("Training model WITHOUT hyperparameter tuning...")
    print("="*60)
    success = categorizer.train(df, tune_hyperparameters=False, save_metrics=True)
    
    if success:
        print("\n✓ Model training completed successfully!")
        print(f"  Model version: {categorizer.version}")
        print(f"  Test accuracy: {categorizer.metrics['accuracy']:.4f}")
        print(f"  F1 Score: {categorizer.metrics['f1_weighted']:.4f}")
    else:
        print("\n✗ Model training failed")
        return
    
    # Optional: Train with hyperparameter tuning (takes longer)
    tune = input("\nTrain with hyperparameter tuning? (y/n): ").lower().strip()
    if tune == 'y':
        print("\n" + "="*60)
        print("Training model WITH hyperparameter tuning...")
        print("="*60)
        categorizer_tuned = MLCategorizer(model_dir='models')
        success_tuned = categorizer_tuned.train(df, tune_hyperparameters=True, save_metrics=True)
        
        if success_tuned:
            print("\n✓ Tuned model training completed!")
            print(f"  Model version: {categorizer_tuned.version}")
            print(f"  Test accuracy: {categorizer_tuned.metrics['accuracy']:.4f}")
            print(f"  F1 Score: {categorizer_tuned.metrics['f1_weighted']:.4f}")
    
    # Test predictions
    print("\n" + "="*60)
    print("Testing predictions...")
    print("="*60)
    
    test_descriptions = [
        "Starbucks Coffee",
        "Walmart Groceries",
        "Netflix Subscription",
        "Uber Ride",
        "Salary Deposit"
    ]
    
    predictions = categorizer.predict(test_descriptions)
    confidences = categorizer.get_confidence(test_descriptions)
    
    print("\nSample Predictions:")
    for desc, pred, conf in zip(test_descriptions, predictions, confidences):
        print(f"  '{desc}' → {pred} (confidence: {conf:.2%})")

if __name__ == "__main__":
    main()
