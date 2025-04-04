"""
Individual Politician Performance Analysis

This script analyzes trading activity of individual politicians,
comparing their performance for two time periods:
- 2024 Q4 (October 1, 2024 - December 30, 2024)
- 2025 Q1 (January 1, 2025 - March 28, 2025)

The analysis calculates:
- Weighted portfolio value changes by politician
- Comparison to market (SPY) performance
- Ranking of top and bottom performers

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
import seaborn as sns
from datetime import datetime
import re
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import math

# Set global plotting style - dark theme with custom colors
plt.style.use('dark_background')

# Define SPY reference prices for calculating market performance
SPY_PRICES = {
    'q1_start': 584.64,  # 1/2/2025
    'q1_end': 563.98     # 3/28/2025
}

# Calculate SPY percent changes
SPY_CHANGES = {
    '2025-Q1': -5.91    # Q1 2025 performance
}

# Define custom color scheme - dark theme with blue/red political colors
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'democrat': '#3C99DC',      # Democrat blue
    'republican': '#E91D0E',    # Republican red
    'spy': '#FFD700',           # Gold for SPY
    'positive': '#00FF00',      # Green for positive returns
    'negative': '#FF0000',      # Red for negative returns
}

# Add INPUT_DIR definition
INPUT_DIR = r"C:\Users\ErikWang\Documents\new_poli_analysis\Politician_Quarterly_Analysis\Input"

def setup_directories():
    """Ensure the Politician_Performance directory exists."""
    politician_dir = os.path.dirname(__file__)
    os.makedirs(politician_dir, exist_ok=True)
    return politician_dir

def setup_plot_style():
    """Setup global plotting style for a dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.color': COLORS['grid'],
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.weight': 'normal',
        'legend.facecolor': COLORS['background'],
        'legend.edgecolor': COLORS['grid'],
        'figure.titlesize': 18,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

def parse_range_values(range_str):
    """
    Parse the 'Range' column to extract the upper dollar amount.
    Example: "$1,001 - $15,000" -> 15000
    """
    if pd.isna(range_str):
        return 0
    
    # Extract numbers from the range string
    numbers = re.findall(r'[$]([0-9,]+)', range_str)
    if not numbers or len(numbers) < 2:
        return 0
    
    # Take the higher number (second value in the range)
    try:
        # Remove commas and convert to float
        return float(numbers[1].replace(',', ''))
    except (ValueError, IndexError):
        return 0

def preprocess_data(df):
    """Preprocess the data with the same logic as party performance"""
    print("Preprocessing data...")
    
    # Convert dates to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
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
    
    # Print verification of calculations
    print("\nVerifying trade result calculations:")
    print("\nSample of sales:")
    sales = df[df['Transaction'].str.contains('sale', case=False)].head(5)
    print(sales[['Transaction', 'PercentChange', 'TradeResult', 'TransactionType']])
    print("\nSample of purchases:")
    purchases = df[~df['Transaction'].str.contains('sale', case=False)].head(5)
    print(purchases[['Transaction', 'PercentChange', 'TradeResult', 'TransactionType']])
    
    # Remove TradeResult and WeightedTradeReturn columns if they exist
    columns_to_drop = ['TradeResult', 'WeightedTradeReturn']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Filter out exchanges and other transaction types
    df = df[df['TransactionType'].isin(['sale', 'purchase'])]
    
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
    
    # Add full party name
    df['PartyName'] = df['Party'].map({'D': 'Democrats', 'R': 'Republicans'})
    
    # Filter for Q1 2025 only
    q1_2025_start = pd.Timestamp('2025-01-01')
    q1_2025_end = pd.Timestamp('2025-03-31')  # Updated to March 31
    
    # Filter for Q1 2025 only
    df = df[
        (df['TransactionDate'] >= q1_2025_start) & 
        (df['TransactionDate'] <= q1_2025_end)
    ]
    df['Quarter'] = '2025-Q1'
    
    print("\nVerifying sale return calculations:")
    sample_sales = df[df['TransactionType'] == 'sale'].head()
    print("\nSample of sale transactions and their returns:")
    print(sample_sales[['TransactionDate', 'Ticker', 'TransactionType', 'PercentChange']])
    
    print("\nData Summary:")
    print(f"Total trades (excluding exchanges): {len(df)}")
    print(f"Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
    print(f"\nTrades by transaction type:")
    print(df['TransactionType'].value_counts())
    print(f"\nTrades by party:")
    print(df['PartyName'].value_counts())
    print(f"\nTotal trade amount: ${df['TradeAmount'].sum():,.2f}")
    
    return df

def calculate_politician_performance(df):
    """Calculate performance metrics by politician"""
    print("\nCalculating politician performance metrics...")
    
    # Calculate politician totals for weighting
    politician_totals = df.groupby(['Quarter', 'Representative'])['TradeAmount'].sum().reset_index()
    total_lookup = {(row['Quarter'], row['Representative']): row['TradeAmount'] 
                    for _, row in politician_totals.iterrows()}
    
    # Calculate weighted returns - SIMPLE LOGIC:
    # Only flip sign for 'Sale (Partial)' or 'Sale (Full)'
    def calculate_weighted_return(row):
        # Check for exact matches only
        if row['Transaction'] in ['Sale (Partial)', 'Sale (Full)']:
            return -1 * (row['PercentChange'] * row['TradeAmount']) / total_lookup[(row['Quarter'], row['Representative'])]
        else:
            return (row['PercentChange'] * row['TradeAmount']) / total_lookup[(row['Quarter'], row['Representative'])]
    
    # Apply the calculation
    df['WeightedReturn'] = df.apply(calculate_weighted_return, axis=1)
    
    # Group by Quarter and Politician
    politician_performance = df.groupby(['Quarter', 'Representative', 'PartyName']).agg({
        'TradeAmount': 'sum',
        'WeightedReturn': 'sum',
        'TransactionType': 'count',
        'Ticker': 'nunique'
    }).reset_index()
    
    # Rename columns for clarity
    politician_performance.columns = ['Quarter', 'Representative', 'PartyName', 
                                   'PortfolioValue', 'WeightedReturn', 
                                   'TotalTrades', 'UniqueStocks']  # Renamed from PurchaseCount to TotalTrades
    
    # Add SPY comparison
    politician_performance['SPYReturn'] = politician_performance['Quarter'].map(SPY_CHANGES)
    politician_performance['ExcessReturn'] = politician_performance['WeightedReturn'] - politician_performance['SPYReturn']
    
    # Print summary statistics
    print("\nSummary of politician returns:")
    print(f"Number of politicians: {len(politician_performance)}")
    print(f"Average return: {politician_performance['WeightedReturn'].mean():.2f}%")
    print(f"Median return: {politician_performance['WeightedReturn'].median():.2f}%")
    print(f"Min return: {politician_performance['WeightedReturn'].min():.2f}%")
    print(f"Max return: {politician_performance['WeightedReturn'].max():.2f}%")
    
    return politician_performance

def plot_politician_performance(performance_data, output_dir, quarter, top_n=None):
    """
    Create a horizontal bar chart showing all politicians' performance for a specific quarter.
    Includes a vertical line for SPY performance.
    
    Parameters:
        performance_data: DataFrame containing politician performance metrics
        output_dir: Directory to save the visualization
        quarter: Quarter to analyze ('2024-Q4' or '2025-Q1')
        top_n: Maximum number of politicians to include (default: None = include all)
    """
    print(f"Creating politician performance visualization for {quarter}...")
    
    # Filter for the selected quarter
    quarter_data = performance_data[performance_data['Quarter'] == quarter].copy()
    total_politicians = len(quarter_data)
    print(f"Found {total_politicians} politicians with trades in {quarter}")
    
    # No filtering based on trade count - include all politicians
    
    # Get the SPY return for this quarter
    spy_return = SPY_CHANGES[quarter]
    
    # Sort by weighted return in DESCENDING order (most profit at top)
    # This means the largest positive values will be first
    quarter_data = quarter_data.sort_values('WeightedReturn', ascending=False)
    
    # Confirm sorting order in logs
    print(f"Top 3 returns after sorting: {quarter_data['WeightedReturn'].head(3).tolist()}")
    print(f"Bottom 3 returns after sorting: {quarter_data['WeightedReturn'].tail(3).tolist()}")
    
    # Get top N performers (if specified)
    if top_n and len(quarter_data) > top_n:
        top_performers = quarter_data.head(top_n).copy()
        print(f"Showing top {top_n} of {len(quarter_data)} politicians by weighted return")
    else:
        top_performers = quarter_data.copy()
        print(f"Showing all {len(quarter_data)} politicians")
    
    # Create the plot - horizontal bar chart (INCREASED SIZE for better legend placement)
    fig_height = max(10, len(top_performers) * 0.45)  # Taller height for each politician
    plt.figure(figsize=(16, fig_height))  # Wider figure to accommodate legend
    
    # CRITICAL FIX: Reverse the order for display so highest returns are at the top
    # We already sorted with highest first, but matplotlib plots from bottom to top
    # So we need to reverse the order for display
    top_performers = top_performers.iloc[::-1]  # Reverse the order
    
    # Create positions for bars
    positions = np.arange(len(top_performers))
    
    # Create color map based on party
    colors = [COLORS['democrat'] if party == 'Democrats' else COLORS['republican'] 
              for party in top_performers['PartyName']]
    
    # Plot horizontal bars
    bars = plt.barh(positions, top_performers['WeightedReturn'], height=0.7, color=colors)
    
    # Add vertical line for SPY performance
    plt.axvline(x=spy_return, color=COLORS['spy'], linestyle='--', linewidth=2, 
                label=f'SPY: {spy_return:.1f}%')
    
    # Add SPY value label on the x-axis
    plt.text(spy_return, -1.5, f"{spy_return:.1f}%", 
             ha='center', va='center', color=COLORS['spy'], fontweight='bold',
             bbox=dict(facecolor=COLORS['background'], edgecolor=COLORS['spy'], boxstyle='round,pad=0.5'))
    
    # Add labels and title
    plt.title(f'Politician Portfolio Performance - {quarter}', fontsize=22, pad=20)
    
    # Move the subtitle below the x-axis label to avoid overlap with title
    plt.xlabel('Percent Change (%)', fontsize=14, labelpad=10)
    plt.figtext(0.5, 0.02, 
                'Each bar shows the weighted percent change in politician stock portfolio', 
                fontsize=12, ha='center', color=COLORS['text'])
    
    # Create party legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['democrat'], label='Democrat'),
        Patch(facecolor=COLORS['republican'], label='Republican'),
        Patch(facecolor=COLORS['spy'], label=f'SPY: {spy_return:.1f}%')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Format y-axis with politician names
    plt.yticks(positions, top_performers['Representative'])
    plt.ylabel('Politician', fontsize=14)
    
    # Calculate padding based on the range of returns
    max_return = top_performers['WeightedReturn'].max()
    min_return = top_performers['WeightedReturn'].min()
    
    # Add more padding to both sides to ensure all labels are visible
    left_padding = 15  # Increased padding for negative side
    right_padding = 25  # Keep existing padding for positive side
    
    # Calculate x-axis limits
    x_min = min(min_return - left_padding, spy_return - 10, -5)
    x_max = max(max_return + right_padding, spy_return + 10)
    
    # Update x-axis limits
    plt.xlim(left=x_min, right=x_max)
    
    # Add gridlines at regular intervals
    grid_step = 10  # Grid lines every 10%
    grid_ticks = np.arange(
        math.floor(x_min / grid_step) * grid_step,
        math.ceil(x_max / grid_step) * grid_step + grid_step,
        grid_step
    )
    plt.xticks(grid_ticks)
    
    # Make the grid more visible
    plt.grid(axis='x', linestyle='--', alpha=0.7, which='major')
    
    # Update label positioning logic to handle large negative values better
    for i, (value, party, formatted_value) in enumerate(zip(
        top_performers['WeightedReturn'], 
        top_performers['PartyName'],
        top_performers['WeightedReturn'].apply(lambda x: f"{x:.1f}%")
    )):
        color = COLORS['text']
        
        # Place labels with more spacing from the bar end
        bar_end_x = value
        if value < 0:
            # For negative values, place label to the left of the bar
            label_x = min(bar_end_x - 1, -2)
            ha = 'right'  # Align text to the right
        else:
            # For positive values, place label to the right of the bar
            label_x = max(bar_end_x + 1, 2)
            ha = 'left'  # Align text to the left
        
        plt.text(label_x, i, formatted_value, va='center', ha=ha, color=color)
    
    # Move portfolio value annotations further right if needed
    portfolio_x = max(x_max - 20, top_performers['WeightedReturn'].max() + 5)
    for i, (value, trades) in enumerate(zip(top_performers['PortfolioValue'], top_performers['TotalTrades'])):  # Changed from PurchaseCount to TotalTrades
        plt.text(portfolio_x, i, 
                f"${value:,.0f} ({trades} trades)", 
                va='center', color=COLORS['text'], fontsize=10)
    
    # Add extra space below the bottom-most bar for the SPY label
    plt.ylim(-2, len(top_performers))
    
    # Add a source watermark (moved to bottom margin)
    plt.figtext(0.5, 0.005, 'Data source: Quiver Quantitative', 
                fontsize=8, ha='center', color=COLORS['text'])
    
    # Save the figure with adjusted layout to make room for subtitle
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_path = os.path.join(output_dir, f'politician_performance_{quarter}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved politician performance visualization to: {output_path}")
    
    return output_path

def save_raw_data(df, output_dir):
    """Save the filtered data to CSV for reference."""
    print("\nSaving cleaned data to CSV files...")
    
    # Save main dataset with weighted trade returns
    main_output = os.path.join(output_dir, 'cleaned_trading_data.csv')
    df.to_csv(main_output, index=False)
    print(f"Saved all trades to: {main_output}")
    
    # Create and save summary by politician
    pivot_by_politician = pd.pivot_table(
        df,
        values=['TradeAmount', 'WeightedReturn'],
        index=['Quarter', 'PartyName', 'Representative'],
        aggfunc={
            'TradeAmount': 'sum',
            'WeightedReturn': 'sum'  # Sum the pre-calculated weighted returns
        }
    ).round(2).reset_index()
    
    politician_output = os.path.join(output_dir, 'summary_by_politician.csv')
    pivot_by_politician.to_csv(politician_output, index=False)
    print(f"Saved politician summary to: {politician_output}")
    
    return main_output

def main():
    """Main function to analyze individual politician performance and create visualizations."""
    print("Starting Individual Politician Performance analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    # Setup output directory
    output_dir = setup_directories()
    
    try:
        # Load data from final_results.csv
        file_path = os.path.join(INPUT_DIR, 'final_results.csv')
        print(f"Loading data from: {file_path}")
        trading_data = pd.read_csv(file_path)
        
        if trading_data is not None:
            # Preprocess data
            df_processed = preprocess_data(trading_data)
            
            if df_processed is not None and not df_processed.empty:
                # Calculate politician performance metrics first
                politician_performance = calculate_politician_performance(df_processed)
                
                # Then save raw filtered data with the weighted returns
                save_raw_data(df_processed, output_dir)
                
                # Save performance data
                performance_csv_path = os.path.join(output_dir, 'politician_performance_data.csv')
                politician_performance.to_csv(performance_csv_path, index=False)
                print(f"Saved politician performance data to: {performance_csv_path}")
                
                # Create visualizations for each quarter - include ALL politicians
                for quarter in ['2025-Q1']:
                    plot_politician_performance(politician_performance, output_dir, quarter)
                
                print("Individual Politician Performance analysis complete. All outputs saved to the Politician_Performance directory.")
            else:
                print("Error: No valid data available for analysis after filtering.")
        else:
            print("Error: Could not load data from final_results.csv")
        
    except Exception as e:
        print(f"Error during Politician Performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 