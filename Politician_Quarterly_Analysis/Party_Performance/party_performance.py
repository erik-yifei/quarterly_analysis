"""
Party Performance Analysis

This script analyzes congressional trading activity by political party (Democrats vs Republicans),
comparing performance for two time periods:
- 2024 Q4 (October 1, 2024 - December 30, 2024)
- 2025 Q1 (January 1, 2025 - March 28, 2025)

The analysis calculates:
- Total portfolio value changes by party
- Comparison to market (SPY) performance
- Weighted percent change visualization

SPY reference prices:
- 10/1/2024: $568.62
- 12/30/2024: $585.19
- 1/2/2025: $584.64
- 3/28/2025: $563.98
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Define input/output paths
INPUT_DIR = r"C:\Users\ErikWang\Documents\new_poli_analysis\Politician_Quarterly_Analysis\Input"
OUTPUT_DIR = os.path.dirname(__file__)

# Define SPY reference prices for calculating market performance
SPY_PRICES = {
    'q4_start': 568.62,  # 10/1/2024
    'q4_end': 585.19,    # 12/30/2024
    'q1_start': 584.64,  # 1/2/2025
    'q1_end': 563.98     # 3/28/2025
}

# Calculate SPY percent changes
SPY_CHANGES = {
    '2024-Q4': 2.21,    # Manually set Q4 2024 performance
    '2025-Q1': -5.91,   # Manually set Q1 2025 performance
    'Combined': -3.84    # Optional: can calculate combined if needed
}

# Define custom color scheme - dark theme with blue/red political colors
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'democrat': '#3C99DC',      # Democrat blue
    'democrat_light': '#5CADEC', # Lighter democrat blue
    'republican': '#E91D0E',    # Republican red
    'republican_light': '#F55B4E', # Lighter republican red
    'spy': '#FFD700',           # Gold for SPY
}

def setup_plot_style():
    """Setup global plotting style for a dark theme with political colors."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'font.family': 'sans-serif'
    })

def load_and_prepare_data():
    """Load and prepare data from final_results.csv"""
    # Load the final results
    file_path = os.path.join(INPUT_DIR, 'final_results.csv')
    df = pd.read_csv(file_path)
    
    # Convert dates to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Filter for the two quarters of interest with updated end dates
    q4_2024_start = pd.Timestamp('2024-10-01')
    q4_2024_end = pd.Timestamp('2024-12-31')  # Changed back to include Dec 31
    q1_2025_start = pd.Timestamp('2025-01-01')
    q1_2025_end = pd.Timestamp('2025-03-28')
    
    conditions = [
        (df['TransactionDate'] >= q4_2024_start) & (df['TransactionDate'] <= q4_2024_end),
        (df['TransactionDate'] >= q1_2025_start) & (df['TransactionDate'] <= q1_2025_end)
    ]
    choices = ['2024-Q4', '2025-Q1']
    df['Quarter'] = np.select(conditions, choices, default='Other')
    
    # Filter out trades not in our quarters of interest
    df = df[df['Quarter'].isin(['2024-Q4', '2025-Q1'])]
    
    # Calculate trade result based on Transaction type
    def calculate_trade_result(row):
        transaction = str(row['Transaction']).lower()
        price_change = row['PercentChange']
        
        # For any type of sale (sale, sale (full), sale (partial))
        if 'sale' in transaction:
            return -price_change  # Profit when price goes down, loss when price goes up
        else:  # purchase
            return price_change   # Profit when price goes up, loss when price goes down
    
    # Add trade result column
    df['TradeResult'] = df.apply(calculate_trade_result, axis=1)
    
    # Clean transaction type for graphing only
    def clean_transaction_type(transaction):
        transaction = str(transaction).lower()
        if 'exchange' in transaction:
            return 'exchange'
        elif any(sale_type in transaction for sale_type in ['sale', 'sold']):
            return 'sale'
        elif any(purchase_type in transaction for purchase_type in ['purchase', 'bought']):
            return 'purchase'
        else:
            return 'other'
    
    # Add cleaned transaction type column (for graphing only)
    df['TransactionType'] = df['Transaction'].apply(clean_transaction_type)
    
    # Filter out exchanges and other transaction types
    df = df[df['TransactionType'].isin(['sale', 'purchase'])]
    
    # Print verification of calculations
    print("\nVerifying trade result calculations:")
    print("\nSample of trades:")
    sample = df.head(10)
    print(sample[['Transaction', 'PercentChange', 'TradeResult', 'TransactionType']])
    
    # Remove TradeResult and WeightedTradeReturn columns if they exist
    columns_to_drop = ['TradeResult', 'WeightedTradeReturn']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Extract upper limit from Range column
    def extract_upper_limit(range_str):
        try:
            # Extract the second number (upper limit) from strings like "$1,001 - $15,000"
            upper_amount = range_str.split('-')[1].strip()
            # Remove '$' and ',' then convert to float
            return float(upper_amount.replace('$', '').replace(',', ''))
        except:
            return 0.0
    
    # Add trade amount column based on Range upper limit
    df['TradeAmount'] = df['Range'].apply(extract_upper_limit)
    
    # Add full party name if not already present
    if 'PartyName' not in df.columns:
        df['PartyName'] = df['Party'].map({'D': 'Democrats', 'R': 'Republicans'})
    
    print("\nData Summary:")
    print(f"Total trades (excluding exchanges): {len(df)}")
    print(f"Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
    print(f"\nTrades by quarter:")
    print(df['Quarter'].value_counts())
    print(f"\nTrades by transaction type:")
    print(df['TransactionType'].value_counts())
    print(f"\nTrades by party:")
    print(df['PartyName'].value_counts())
    print(f"\nTotal trade amount: ${df['TradeAmount'].sum():,.2f}")
    
    return df

def calculate_party_performance(df):
    """Calculate performance metrics by party"""
    print("\nCalculating party performance metrics...")
    
    # Calculate party totals for weighting
    party_totals = df.groupby(['Quarter', 'PartyName'])['TradeAmount'].sum().reset_index()
    total_lookup = {(row['Quarter'], row['PartyName']): row['TradeAmount'] 
                    for _, row in party_totals.iterrows()}
    
    # Calculate weighted returns - SIMPLE LOGIC:
    # Only flip sign for 'Sale (Partial)' or 'Sale (Full)'
    def calculate_weighted_return(row):
        # Check for exact matches only
        if row['Transaction'] in ['Sale (Partial)', 'Sale (Full)']:
            return -1 * (row['PercentChange'] * row['TradeAmount']) / total_lookup[(row['Quarter'], row['PartyName'])]
        else:
            return (row['PercentChange'] * row['TradeAmount']) / total_lookup[(row['Quarter'], row['PartyName'])]
    
    # Apply the calculation
    df['WeightedReturn'] = df.apply(calculate_weighted_return, axis=1)
    
    # Group by Quarter and Party
    party_performance = df.groupby(['Quarter', 'PartyName']).agg({
        'TradeAmount': 'sum',
        'WeightedReturn': 'sum',
        'TransactionType': 'count',
        'Representative': 'nunique',
        'Ticker': 'nunique'
    }).reset_index()
    
    # Rename columns for clarity
    party_performance.columns = ['Quarter', 'PartyName', 'PortfolioValue', 'WeightedReturn', 
                               'PurchaseCount', 'UniquePoliticians', 'UniqueStocks']
    
    # Add SPY comparison
    party_performance['SPYReturn'] = party_performance['Quarter'].map(SPY_CHANGES)
    party_performance['ExcessReturn'] = party_performance['WeightedReturn'] - party_performance['SPYReturn']
    
    # Print detailed summary
    print("\nParty Performance Summary:")
    for _, row in party_performance.iterrows():
        print(f"\n{row['PartyName']} in {row['Quarter']}:")
        print(f"  Portfolio Value: ${row['PortfolioValue']:,.2f}")
        print(f"  Weighted Return: {row['WeightedReturn']:.2f}%")
        print(f"  SPY Return: {row['SPYReturn']:.2f}%")
        print(f"  Excess Return: {row['ExcessReturn']:.2f}%")
        print(f"  {row['UniquePoliticians']} politicians trading {row['UniqueStocks']} unique stocks")
    
    return party_performance

def plot_party_performance(performance_data):
    """Create visualizations for party performance"""
    print("\nCreating performance visualizations...")
    
    for quarter in performance_data['Quarter'].unique():
        quarter_data = performance_data[performance_data['Quarter'] == quarter].copy()
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        parties = quarter_data['PartyName'].tolist()
        returns = quarter_data['WeightedReturn'].tolist()
        colors = [COLORS['democrat'] if p == 'Democrats' else COLORS['republican'] for p in parties]
        
        # Add SPY comparison
        spy_return = SPY_CHANGES[quarter]
        parties.append('SPY')
        returns.append(spy_return)
        colors.append(COLORS['spy'])
        
        # Plot bars
        plt.barh(parties, returns, height=0.6, color=colors)
        
        # Add labels and styling
        plt.title(f'Portfolio Returns by Party vs SPY - {quarter}', pad=20)
        plt.xlabel('Return (%)')
        
        # Add grid with dotted lines
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Set x-axis limits based on data
        min_return = min(returns)
        max_return = max(returns)
        padding = 1.0  # Padding for labels
        
        # Set symmetric limits around zero with padding
        plt.xlim(min_return - padding, max_return + padding)
        
        # Add value labels with consistent box styling for all values
        for i, v in enumerate(returns):
            if v < 0:
                # For negative values, place label to the left of bar end
                x_pos = v - 0.2
                ha = 'right'
            else:
                # For positive values, place label to the right of bar end
                x_pos = v + 0.2
                ha = 'left'
            
            plt.text(x_pos, i, f'{v:.1f}%', 
                    va='center',
                    ha=ha,
                    color=COLORS['text'],
                    bbox=dict(
                        facecolor=COLORS['background'],
                        edgecolor=COLORS['grid'],
                        boxstyle='round,pad=0.5',
                        alpha=1.0
                    ))
        
        # Save plot
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'party_performance_{quarter}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        plt.close()
        
        print(f"Saved visualization for {quarter}")

def save_cleaned_data(df, output_dir):
    """Save the cleaned and processed data to CSV files"""
    print("\nSaving cleaned data to CSV files...")
    
    # Update raw columns list to only include columns we have
    raw_columns = [
        'Quarter',
        'TransactionDate',
        'Representative',
        'PartyName',
        'Ticker',
        'Transaction',  # Changed from TransactionType to Transaction
        'Range',
        'TradeAmount',
        'EntryPrice',
        'ExitPrice',
        'PercentChange',
        'WeightedReturn'  # Changed from WeightedTradeReturn
    ]
    
    # Save raw data with all calculations
    raw_output = os.path.join(output_dir, 'raw_trading_data.csv')
    df[raw_columns].to_csv(raw_output, index=False)
    print(f"\nSaved raw trading data with calculations to: {raw_output}")
    
    # Print sample of raw data for verification
    print("\nSample of raw trading data (first few trades):")
    sample_df = df[raw_columns].head()
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Don't wrap
    print(sample_df.to_string())
    
    # Print summary statistics for verification
    print("\nSummary Statistics by Party and Quarter:")
    summary = df.groupby(['Quarter', 'PartyName']).agg({
        'TradeAmount': ['sum', 'mean', 'count'],
        'WeightedReturn': 'sum',
        'Representative': 'nunique',
        'PercentChange': ['mean', 'min', 'max'],
        'EntryPrice': ['mean', 'min', 'max'],
        'ExitPrice': ['mean', 'min', 'max']
    }).round(2)
    print(summary.to_string())
    
    # Save party-level summary
    pivot_by_party = pd.pivot_table(
        df,
        values=['TradeAmount', 'WeightedReturn', 'PercentChange'],
        index=['Quarter', 'PartyName'],
        aggfunc={
            'TradeAmount': 'sum',
            'WeightedReturn': 'sum',
            'PercentChange': ['mean', 'count']
        }
    ).round(2).reset_index()
    
    party_output = os.path.join(output_dir, 'summary_by_party.csv')
    pivot_by_party.to_csv(party_output, index=False)
    print(f"\nSaved party summary to: {party_output}")
    
    # Save representative-level summary
    pivot_by_rep = pd.pivot_table(
        df,
        values=['TradeAmount', 'WeightedReturn', 'PercentChange'],
        index=['Quarter', 'PartyName', 'Representative'],
        aggfunc={
            'TradeAmount': 'sum',
            'WeightedReturn': 'sum',
            'PercentChange': ['mean', 'count']
        }
    ).round(2).reset_index()
    
    rep_output = os.path.join(output_dir, 'summary_by_representative.csv')
    pivot_by_rep.to_csv(rep_output, index=False)
    print(f"\nSaved representative summary to: {rep_output}")
    
    # Print verification of total weighted returns
    print("\nVerification of Total Weighted Returns by Party and Quarter:")
    verification = df.groupby(['Quarter', 'PartyName']).agg({
        'WeightedReturn': 'sum',
        'PercentChange': 'mean'
    }).round(2)
    print(verification.to_string())
    
    return raw_output

def main():
    """Main function to analyze party performance"""
    print("Starting Party Performance Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        if df is not None and not df.empty:
            # Calculate performance metrics first (this will add WeightedTradeReturn column)
            performance_data = calculate_party_performance(df)
            
            # Now save cleaned data with the new weighted returns
            save_cleaned_data(df, OUTPUT_DIR)
            
            # Create visualizations
            plot_party_performance(performance_data)
            
            # Save performance summary
            output_path = os.path.join(OUTPUT_DIR, 'party_performance_summary.csv')
            performance_data.to_csv(output_path, index=False)
            print(f"\nSaved performance summary to: {output_path}")
            
        else:
            print("Error: No valid data available for analysis.")
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 