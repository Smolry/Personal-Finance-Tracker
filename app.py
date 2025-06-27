import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from src.data_loader import load_csv_transactions, save_transactions_to_sqlite
from src.categorizer import categorize_transactions, train_categorizer
from src.insights import analyze_spending_trends, detect_recurring_expenses, suggest_budget

st.set_page_config(page_title="Personal Finance Manager", layout="wide")
st.title("Personal Finance Manager")

# Initialize session state for transactions
if 'transactions' not in st.session_state:
    st.session_state.transactions = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Dashboard", "Insights", "Budget Planner"])

# Upload Data page
if page == "Upload Data":
    st.header("Upload Transaction Data")
    
    uploaded_file = st.file_uploader("Upload your transaction CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and process transactions
        transactions = load_csv_transactions(uploaded_file)
        transactions = categorize_transactions(transactions)
        
        # Save to session state
        st.session_state.transactions = transactions
        
        # Display raw data
        st.subheader("Transaction Data")
        st.dataframe(transactions)
        
        # Save to database
        if st.button("Save transactions to database"):
            success = save_transactions_to_sqlite(transactions)
            if success:
                st.success("Transactions saved successfully!")
                
        # Train ML model
        if st.button("Train categorization model with this data"):
            success = train_categorizer(transactions)
            if success:
                st.success("Categorization model trained successfully!")
            else:
                st.warning("Not enough categorized transactions to train the model.")

# Dashboard page
elif page == "Dashboard":
    st.header("Financial Dashboard")
    
    if st.session_state.transactions is None:
        st.info("Please upload transaction data first.")
    else:
        transactions = st.session_state.transactions
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        # Total Income
        total_income = transactions[transactions['amount'] > 0]['amount'].sum()
        col1.metric("Total Income", f"${total_income:.2f}")
        
        # Total Expenses
        total_expenses = abs(transactions[transactions['amount'] < 0]['amount'].sum())
        col2.metric("Total Expenses", f"${total_expenses:.2f}")
        
        # Net Savings
        net_savings = total_income - total_expenses
        col3.metric("Net Savings", f"${net_savings:.2f}")
        
        # Basic visualizations
        st.subheader("Spending by Category")
        
        # Filter for expenses only (negative amounts)
        expenses = transactions[transactions['amount'] < 0].copy()
        expenses['amount'] = expenses['amount'].abs()  # Make positive for charting
        
        # Group by category and sum
        category_spending = expenses.groupby('category')['amount'].sum().reset_index()
        category_spending = category_spending.sort_values('amount', ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='amount', y='category', data=category_spending, ax=ax)
        ax.set_title('Spending by Category')
        ax.set_xlabel('Amount Spent')
        ax.set_ylabel('Category')
        
        st.pyplot(fig)
        
        # Monthly spending trend
        st.subheader("Monthly Spending Trend")
        
        # Add month column
        expenses['month'] = expenses['date'].dt.strftime('%Y-%m')
        
        # Group by month and sum
        monthly_spending = expenses.groupby('month')['amount'].sum().reset_index()
        
        # Create line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='month', y='amount', data=monthly_spending, marker='o', ax=ax)
        ax.set_title('Monthly Spending Trend')
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount Spent')
        plt.xticks(rotation=45)
        
        st.pyplot(fig)

# Insights page
elif page == "Insights":
    st.header("Financial Insights")
    
    if st.session_state.transactions is None:
        st.info("Please upload transaction data first.")
    else:
        transactions = st.session_state.transactions
        
        # Get insights
        insights = analyze_spending_trends(transactions)
        recurring = detect_recurring_expenses(transactions)
        
        # Display top merchants
        st.subheader("Top Merchants")
        st.dataframe(insights['top_merchants'])
        
        # Display recurring expenses
        st.subheader("Recurring Expenses")
        if len(recurring) > 0:
            st.dataframe(recurring[['merchant', 'avg_amount', 'frequency_days', 'next_expected']])
            
            # Visualization of recurring expenses
            fig, ax = plt.subplots(figsize=(10, 6))
            recurring_plot = recurring.sort_values('avg_amount', ascending=False).head(10)
            sns.barplot(x='avg_amount', y='merchant', data=recurring_plot, ax=ax)
            ax.set_title('Top Recurring Expenses')
            ax.set_xlabel('Average Amount')
            ax.set_ylabel('Merchant')
            
            st.pyplot(fig)
        else:
            st.info("No recurring expenses detected.")

# Budget Planner page
elif page == "Budget Planner":
    st.header("Budget Planner")
    
    if st.session_state.transactions is None:
        st.info("Please upload transaction data first.")
    else:
        transactions = st.session_state.transactions
        
        # Get suggested budget
        budget = suggest_budget(transactions)
        
        st.subheader("Suggested Monthly Budget")
        st.dataframe(budget)
        
        # Budget visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        budget_plot = budget.sort_values('budget', ascending=False)
        sns.barplot(x='budget', y='category', data=budget_plot, ax=ax)
        ax.set_title('Suggested Monthly Budget by Category')
        ax.set_xlabel('Budget Amount')
        ax.set_ylabel('Category')
        
        st.pyplot(fig)
        
        # Allow user to adjust budget
        st.subheader("Adjust Your Budget")
        
        # Create a form for budget adjustments
        with st.form("budget_form"):
            adjusted_budget = {}
            
            for _, row in budget.iterrows():
                category = row['category']
                suggested = row['budget']
                adjusted_budget[category] = st.number_input(
                    f"{category} Budget", 
                    min_value=0.0, 
                    value=float(suggested),
                    format="%.2f"
                )
            
            submit = st.form_submit_button("Save Budget")
            
            if submit:
                # In a real app, you would save this to a database
                st.success("Budget saved successfully!")
                
                # Show comparison
                budget_comparison = []
                for category, amount in adjusted_budget.items():
                    suggested = budget[budget['category'] == category]['budget'].values[0]
                    budget_comparison.append({
                        'category': category,
                        'suggested': suggested,
                        'adjusted': amount,
                        'difference': amount - suggested
                    })
                
                comparison_df = pd.DataFrame(budget_comparison)
                st.dataframe(comparison_df)