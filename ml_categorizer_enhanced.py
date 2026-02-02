"""
Enhanced ML Categorizer with evaluation, tuning, and versioning
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MLCategorizer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.pipeline = None
        self.metrics = {}
        self.version = None
        
    def train(self, transactions_df, tune_hyperparameters=False, save_metrics=True):
        """
        Train ML model with optional hyperparameter tuning
        
        Args:
            transactions_df: DataFrame with 'description' and 'category' columns
            tune_hyperparameters: Whether to perform grid search
            save_metrics: Whether to save evaluation metrics
        """
        # Filter out uncategorized transactions
        train_df = transactions_df[transactions_df['category'] != 'Uncategorized'].copy()
        
        if len(train_df) < 50:
            print(f"Not enough data to train: only {len(train_df)} categorized transactions")
            return False
        
        print(f"Training on {len(train_df)} transactions across {train_df['category'].nunique()} categories")
        
        # Create features and target
        X = train_df['description']
        y = train_df['category']
        
        # Split data (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if tune_hyperparameters:
            print("\nPerforming hyperparameter tuning...")
            self.pipeline = self._tune_hyperparameters(X_train, y_train)
        else:
            # Create pipeline with good default parameters
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=500
                )),
                ('clf', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
            
            print("\nTraining model...")
            self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        print("\nEvaluating model...")
        self.metrics = self._evaluate_model(X_train, X_test, y_train, y_test)
        
        # Generate version string
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.model_dir, f'categorizer_v{self.version}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"\nModel saved: {model_path}")
        
        # Save latest version reference
        with open(os.path.join(self.model_dir, 'latest_version.txt'), 'w') as f:
            f.write(self.version)
        
        # Save metrics
        if save_metrics:
            metrics_path = os.path.join(self.model_dir, f'metrics_v{self.version}.json')
            with open(metrics_path, 'w') as f:
                # Convert numpy types and pandas Series to native Python types for JSON
                metrics_json = {}
                for k, v in self.metrics.items():
                    if isinstance(v, np.ndarray):
                        metrics_json[k] = v.tolist()
                    elif isinstance(v, (np.float32, np.float64)):
                        metrics_json[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        metrics_json[k] = int(v)
                    elif isinstance(v, pd.Series):
                        metrics_json[k] = v.tolist()
                    elif k in ['y_pred', 'y_test', 'confusion_matrix', 'classification_report']:
                        # Skip these for JSON - they're too large or complex
                        continue
                    else:
                        metrics_json[k] = v
                json.dump(metrics_json, f, indent=2)
            print(f"Metrics saved: {metrics_path}")
            
            # Generate and save visualizations
            self._save_visualizations(y_test, self.metrics['y_pred'])
        
        return True
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Perform grid search for hyperparameter tuning"""
        param_grid = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [300, 500],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [15, 20, None],
            'clf__min_samples_split': [2, 5]
        }
        
        base_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, max_df=0.95)),
            ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_train_pred = self.pipeline.predict(X_train)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train, cv=5, scoring='f1_weighted'
        )
        
        # Feature importance (top keywords per category)
        feature_importance = self._extract_feature_importance()
        
        metrics = {
            'accuracy': float(accuracy),
            'train_accuracy': float(train_accuracy),
            'precision_weighted': float(precision_w),
            'recall_weighted': float(recall_w),
            'f1_weighted': float(f1_w),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_test': y_test,
            'classification_report': report,
            'feature_importance': feature_importance,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'categories': sorted(y_test.unique().tolist())
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Training samples:     {metrics['n_train']}")
        print(f"Test samples:         {metrics['n_test']}")
        print(f"Test Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Train Accuracy:       {metrics['train_accuracy']:.4f}")
        print(f"Weighted F1 Score:    {metrics['f1_weighted']:.4f}")
        print(f"CV F1 Score:          {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"\nPer-Category Performance:")
        print(f"{'-'*60}")
        for category in metrics['categories']:
            cat_metrics = report[category]
            print(f"{category:20s} | F1: {cat_metrics['f1-score']:.3f} | "
                  f"Precision: {cat_metrics['precision']:.3f} | "
                  f"Recall: {cat_metrics['recall']:.3f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract top keywords for each category"""
        if self.pipeline is None:
            return {}
        
        # Get the vectorizer and classifier
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['clf']
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importances from Random Forest
        importances = classifier.feature_importances_
        
        # Get class labels
        classes = classifier.classes_
        
        # For each class, get top features
        top_features_per_class = {}
        
        # Get top features overall
        top_indices = np.argsort(importances)[-30:][::-1]
        top_features = [(feature_names[i], float(importances[i])) for i in top_indices]
        
        return {
            'top_features_overall': top_features,
            'categories': classes.tolist()
        }
    
    def _save_visualizations(self, y_test, y_pred):
        """Generate and save evaluation visualizations"""
        viz_dir = os.path.join(self.model_dir, f'visualizations_v{self.version}')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        categories = sorted(y_test.unique())
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories
        )
        plt.title(f'Confusion Matrix - Model v{self.version}', fontsize=14, fontweight='bold')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # 2. Per-Category Performance
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        categories = [cat for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics_df = pd.DataFrame({
            'Category': categories,
            'Precision': [report[cat]['precision'] for cat in categories],
            'Recall': [report[cat]['recall'] for cat in categories],
            'F1-Score': [report[cat]['f1-score'] for cat in categories]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Category Performance - Model v{self.version}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'category_performance.png'), dpi=150)
        plt.close()
        
        print(f"Visualizations saved to {viz_dir}/")
    
    def load_model(self, version=None):
        """Load a trained model (latest by default)"""
        if version is None:
            # Load latest version
            version_file = os.path.join(self.model_dir, 'latest_version.txt')
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    version = f.read().strip()
            else:
                # Fallback to old model path for backwards compatibility
                old_path = 'data/categorizer_model.pkl'
                if os.path.exists(old_path):
                    with open(old_path, 'rb') as f:
                        self.pipeline = pickle.load(f)
                    return True
                return False
        
        model_path = os.path.join(self.model_dir, f'categorizer_v{version}.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            self.version = version
            return True
        return False
    
    def predict(self, descriptions):
        """Predict categories for transaction descriptions"""
        if self.pipeline is None:
            if not self.load_model():
                return ["Uncategorized"] * len(descriptions) if isinstance(descriptions, list) else ["Uncategorized"]
        
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        return self.pipeline.predict(descriptions)
    
    def predict_proba(self, descriptions):
        """Get prediction probabilities"""
        if self.pipeline is None:
            if not self.load_model():
                return None
        
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        return self.pipeline.predict_proba(descriptions)
    
    def get_confidence(self, descriptions):
        """Get confidence scores for predictions"""
        probas = self.predict_proba(descriptions)
        if probas is None:
            return None
        
        # Get max probability for each prediction
        confidences = np.max(probas, axis=1)
        return confidences
