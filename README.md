#  Personal Finance Tracker with ML-Powered Categorization

An intelligent personal finance management system that automatically categorizes transactions using machine learning, detects recurring expenses, and provides data-driven budget recommendations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io)

## 🔗 Live Demo

 **[Try it on Streamlit Cloud](https://personal-finance-tracker-smolry.streamlit.app/)**

---

## ✨ Key Features

###  Machine Learning
- **Automated Transaction Categorization** using Random Forest classifier
- **TF-IDF Feature Engineering** with bi-gram analysis  
- **Model Versioning** with timestamp-based tracking
- **Comprehensive Evaluation Metrics** (accuracy, precision, recall, F1)
- **Confidence Scoring** for active learning
- **Hyperparameter Tuning** via GridSearchCV

###  Financial Analytics
-  **Visual Spending Insights** with interactive charts
-  **Recurring Transaction Detection** (subscriptions, bills)
-  **Smart Budget Recommendations** based on historical patterns
-  **Monthly/Category-wise Breakdowns**
-  **SQLite Persistence** for data storage

### 🎯 ML Performance
- **73.3% Test Accuracy** on 1600+ transaction dataset
- **75.8% Weighted F1 Score**
- **Cross-Validation Score: 76.8%** (±3.3%)
- **12 Transaction Categories** (Dining, Groceries, Utilities, etc.)

---

## 📁 Project Structure

```
Personal-Finance-Tracker/
├── app.py                          # Streamlit web interface
├── src/
│   ├── categorizer.py              # Categorization logic
│   ├── ml_categorizer_enhanced.py  # Enhanced ML model with evaluation
│   ├── data_loader.py              # Data ingestion
│   └── insights.py                 # Analytics & insights
├── models/                         # ML model artifacts
│   ├── categorizer_v*.pkl          # Versioned models
│   ├── metrics_v*.json             # Performance metrics
│   └── visualizations_v*/          # Confusion matrices & plots
├── data/
│   ├── raw/                        # Raw transaction CSVs
│   ├── processed/                  # Cleaned data
│   └── finance.db                  # SQLite database
├── notebooks/
│   └── ML_Pipeline_Finance_Tracker.ipynb  # Full ML workflow
├── scripts/
│   ├── generate_large_dataset.py   # Synthetic data generator
│   └── train_model.py              # Model training script
├── requirements.txt
└── README.md
```

---

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Smolry/Personal-Finance-Tracker.git
cd Personal-Finance-Tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Training Your Own Model

```bash
# Generate synthetic dataset (1600+ transactions)
python scripts/generate_large_dataset.py

# Train the ML model
python scripts/train_model.py
```

---

##  Machine Learning Pipeline

### 1. Feature Engineering
- **TF-IDF Vectorization** with ngram_range=(1,2)
- Captures both single words ("netflix") and phrases ("gas station")
- Min document frequency: 2 (reduces noise)
- Max document frequency: 95% (removes common words)

### 2. Model Architecture
```python
Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=20))
])
```

**Why Random Forest?**
- Handles high-dimensional sparse data (TF-IDF features)
- Robust to overfitting
- Provides feature importance (top keywords per category)
- No feature scaling required
- Works well with imbalanced classes

### 3. Training & Evaluation
- **Train/Test Split:** 80/20 stratified
- **Cross-Validation:** 5-fold CV with F1 scoring
- **Hyperparameter Tuning:** GridSearchCV across 48 combinations
- **Metrics Tracked:**
  - Accuracy, Precision, Recall, F1 (per-category & weighted)
  - Confusion matrix
  - Feature importance
  - Training vs. test performance gap

### 4. Model Versioning
Each trained model gets:
- Timestamp-based version ID
- Serialized .pkl file
- JSON metrics file
- Visualization folder (confusion matrix, category performance)

---

## 📊 Performance Metrics

### Latest Model (v20260202)

| Metric | Score |
|--------|-------|
| Test Accuracy | 73.3% |
| Weighted F1 | 75.8% |
| 5-Fold CV F1 | 76.8% (±3.3%) |
| Training Set | 1,300 transactions |
| Test Set | 326 transactions |

### Per-Category Performance

| Category | F1 Score | Precision | Recall |
|----------|----------|-----------|--------|
| Housing | 100.0% | 100.0% | 100.0% |
| Income | 96.7% | 100.0% | 93.5% |
| Utilities | 95.1% | 100.0% | 90.6% |
| Education | 92.7% | 100.0% | 86.4% |
| Subscriptions | 90.9% | 100.0% | 83.3% |
| Travel | 89.3% | 100.0% | 80.6% |
| Health | 85.1% | 95.2% | 76.9% |
| Entertainment | 77.8% | 100.0% | 63.6% |

**Key Observations:**
- Perfect performance on Housing, Income, Utilities (high-signal keywords)
- Lower recall on Dining/Groceries (similar merchants, need more data)
- Shopping category needs improvement (too generic)

---

##  Jupyter Notebook - Full ML Pipeline

Check out [`ML_Pipeline_Finance_Tracker.ipynb`](notebooks/ML_Pipeline_Finance_Tracker.ipynb) for:
- Exploratory Data Analysis
- Feature engineering decisions
- Model selection rationale
- Hyperparameter tuning process
- Error analysis
- Feature importance visualization

Perfect for presenting in ML interviews!

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Framework** | scikit-learn 1.7+ |
| **Feature Engineering** | TfidfVectorizer |
| **Model** | Random Forest (200 trees) |
| **Web Framework** | Streamlit |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data Processing** | pandas, numpy |

---

## 📊 Dataset

### Synthetic Data Generation
- **1,626 transactions** over 12 months
- **12 categories**: Dining, Groceries, Transportation, Utilities, Entertainment, Shopping, Health, Housing, Travel, Education, Subscriptions, Income
- **Realistic patterns**: Recurring bills, salary deposits, random expenses
- **Merchant diversity**: 100+ unique merchant names

### Real-World Data Support
Accepts CSV files with columns:
- `Date` (any datetime format)
- `Description` (merchant name)
- `Amount` (positive = income, negative = expense)

---
