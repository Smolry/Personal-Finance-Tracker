import pandas as pd
import numpy as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

class MLCategorizer:
    def __init__(self, model_path='data/categorizer_model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        
    def train(self, transactions_df):
        """Train a machine learning model to categorize transactions based on description"""
        # Filter out transactions with 'Uncategorized' category
        train_df = transactions_df[transactions_df['category'] != 'Uncategorized'].copy()
        
        if len(train_df) < 10:  # Need enough samples to train
            print("Not enough categorized transactions to train the model")
            return False
            
        # Create features and target
        X = train_df['description']
        y = train_df['category']
        
        # Create a pipeline with TF-IDF and Random Forest
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
            
        return True
    
    def load_model(self):
        """Load a trained model if it exists"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            return True
        return False
    
    def predict(self, descriptions):
        """Predict categories for a list of transaction descriptions"""
        if self.pipeline is None:
            if not self.load_model():
                return ["Uncategorized"] * len(descriptions)
        
        # Convert single string to list if necessary
        if isinstance(descriptions, str):
            descriptions = [descriptions]
            
        return self.pipeline.predict(descriptions)