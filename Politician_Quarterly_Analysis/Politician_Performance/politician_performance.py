"""
Individual Politician Performance Analysis

This script analyzes trading activity of individual politicians,
comparing their performance for two time periods:
- 2024 Q4 (October 1, 2024 - December 31, 2024)
- 2025 Q1 (January 1, 2025 - March 21, 2025)

The analysis calculates:
- Weighted portfolio value changes by politician
- Comparison to market (SPY) performance
- Ranking of top and bottom performers

SPY reference prices:
- 10/1/2024: $568.62
- 12/31/2024: $586.08
- 1/2/2025: $584.64
- 3/21/2025: $563.98
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import quiverquant
from dotenv import load_dotenv
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
QUIVER_API_KEY = os.getenv("API_KEY")

# Set global plotting style - dark theme with custom colors
plt.style.use('dark_background')

# Define SPY reference prices for calculating market performance
SPY_PRICES = {
    'q4_start': 568.62,  # 10/1/2024
    'q4_end': 586.08,    # 12/31/2024
    'q1_start': 584.64,  # 1/2/2025
    'q1_end': 563.98     # 3/21/2025
}

# Calculate SPY percent changes
SPY_CHANGES = {
    '2024-Q4': (SPY_PRICES['q4_end'] - SPY_PRICES['q4_start']) / SPY_PRICES['q4_start'] * 100,
    '2025-Q1': (SPY_PRICES['q1_end'] - SPY_PRICES['q1_start']) / SPY_PRICES['q1_start'] * 100
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

def get_congress_trading_data():
    """
    Fetches congressional trading data from Quiver Quantitative API.
    
    Returns:
        pandas.DataFrame: DataFrame containing all available congressional trading data
    """
    try:
        # Initialize the Quiver client with API key
        print(f"Initializing Quiver client with API key: {QUIVER_API_KEY[:5] if QUIVER_API_KEY else None}..." + "*" * 10)
        quiver = quiverquant.quiver(QUIVER_API_KEY)
        
        # Fetch trading data
        print("Fetching congressional trading data...")
        congress_data = quiver.congress_trading()
        
        print(f"Retrieved {len(congress_data)} trading records")
        return congress_data
    
    except Exception as e:
        print(f"Error fetching data from Quiver Quantitative API: {str(e)}")
        return None

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
    """
    Preprocess the data:
    1. Convert dates to datetime
    2. Extract trade values from Range column
    3. Filter for the specific quarters
    4. Filter for Democrat and Republican only
    5. Standardize transaction types (combine all sale types)
    6. Filter out all Exchange transactions
    """
    print("Preprocessing data...")
    # Convert dates to datetime
    df['Date'] = pd.to_datetime(df['TransactionDate'])
    
    # Extract trade values from Range column
    df['TradeValueUpper'] = df['Range'].apply(parse_range_values)
    
    # Define date ranges for quarters
    q4_2024_start = pd.Timestamp('2024-10-01')
    q4_2024_end = pd.Timestamp('2024-12-31')
    q1_2025_start = pd.Timestamp('2025-01-01')
    q1_2025_end = pd.Timestamp('2025-03-21')  # Using the date from SPY reference
    
    # Create quarter indicator
    conditions = [
        (df['Date'] >= q4_2024_start) & (df['Date'] <= q4_2024_end),
        (df['Date'] >= q1_2025_start) & (df['Date'] <= q1_2025_end)
    ]
    choices = ['2024-Q4', '2025-Q1']
    df['Quarter'] = np.select(conditions, choices, default='Other')
    
    # Filter for the two quarters of interest
    df_filtered = df[df['Quarter'].isin(['2024-Q4', '2025-Q1'])].copy()
    
    if df_filtered.empty:
        print("Warning: No data found for the specified quarters (2024-Q4 and 2025-Q1)")
        return None
    
    # Filter for Democrats and Republicans only (exclude Independents)
    original_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['Party'].isin(['D', 'R'])].copy()
    dem_rep_count = len(df_filtered)
    print(f"Filtered to Democrats and Republicans only: {dem_rep_count} records (removed {original_count - dem_rep_count} records)")
    
    # Add full party name for better readability
    df_filtered['PartyName'] = df_filtered['Party'].map({'D': 'Democrats', 'R': 'Republicans'})
    
    # Standardize transaction types - combine all sale types into a single "Sale" category
    print("Standardizing transaction types...")
    
    # Print unique transaction types before standardization
    unique_tx_types = df_filtered['Transaction'].unique()
    print(f"Original transaction types: {unique_tx_types}")
    
    # Create standardized transaction type
    def standardize_transaction(tx_type):
        tx_type = str(tx_type).lower()
        if 'purchase' in tx_type:
            return 'Purchase'
        elif any(sale_term in tx_type for sale_term in ['sale', 'sold']):
            return 'Sale'
        elif 'exchange' in tx_type:
            return 'Exchange'
        else:
            return tx_type.capitalize()
    
    df_filtered['StandardizedTransaction'] = df_filtered['Transaction'].apply(standardize_transaction)
    
    # Print unique transaction types after standardization
    unique_std_tx_types = df_filtered['StandardizedTransaction'].unique()
    print(f"Standardized transaction types: {unique_std_tx_types}")
    
    # Count transactions by standardized type
    tx_counts = df_filtered['StandardizedTransaction'].value_counts()
    print("\nTransaction counts after standardization:")
    for tx_type, count in tx_counts.items():
        print(f"  {tx_type}: {count}")
    
    # Filter out Exchange transactions
    before_exchange_filter = len(df_filtered)
    df_filtered = df_filtered[df_filtered['StandardizedTransaction'] != 'Exchange'].copy()
    after_exchange_filter = len(df_filtered)
    exchange_removed = before_exchange_filter - after_exchange_filter
    print(f"\nRemoved {exchange_removed} Exchange transactions (filtered from {before_exchange_filter} to {after_exchange_filter} records)")
    
    # Convert PriceChange and SPYChange to numeric if they're not already
    if 'PriceChange' in df_filtered.columns:
        df_filtered['PriceChange'] = pd.to_numeric(df_filtered['PriceChange'], errors='coerce')
    
    if 'SPYChange' in df_filtered.columns:
        df_filtered['SPYChange'] = pd.to_numeric(df_filtered['SPYChange'], errors='coerce')
    
    print(f"Final filtered dataset: {len(df_filtered)} records (D/R parties, Purchase/Sale only, Q4 2024 and Q1 2025)")
    return df_filtered

def calculate_politician_performance(df):
    """
    Calculate performance metrics by politician, accounting for trade direction.
    
    Steps:
    1. Adjust price change based on transaction type (Purchase vs Sale)
    2. Calculate weighted returns for each politician in each quarter
    3. Compare to SPY performance
    """
    print("Calculating politician performance metrics...")
    
    # STEP 1: Create adjusted price change accounting for transaction type
    # For purchases: positive price change = profit
    # For sales: negative price change = profit (reversed sign)
    df['AdjustedPriceChange'] = df.apply(
        lambda row: row['PriceChange'] if row['StandardizedTransaction'] == 'Purchase' else -row['PriceChange'], 
        axis=1
    )
    
    # Make sure the sign reversal for sales was applied correctly
    sale_count = len(df[df['StandardizedTransaction'] == 'Sale'])
    purchase_count = len(df[df['StandardizedTransaction'] == 'Purchase'])
    print(f"Adjusted price change sign for {sale_count} sale transactions")
    print(f"Dataset contains {purchase_count} purchases and {sale_count} sales")
    
    # Function to calculate weighted average return for a politician's trades
    def weighted_average(group):
        # Extract trade values (weights) and adjusted returns (values)
        weights = group['TradeValueUpper']  # Dollar amount of each trade
        values = group['AdjustedPriceChange']  # Direction-adjusted price change percentage
        
        # Filter out NaN values from both arrays
        mask = ~values.isna()
        weights, values = weights[mask], values[mask]
        
        # If no valid data, return NaN
        if len(weights) == 0 or weights.sum() == 0:
            return np.nan
        
        # Calculate weighted sum directly for more control
        weighted_sum = np.sum(weights * values)
        total_weight = np.sum(weights)
        
        # Calculate weighted average and return full precision
        return weighted_sum / total_weight
    
    # Enable validation for random politicians to verify calculation correctness
    VALIDATION_SAMPLE_SIZE = 5  # Number of politicians to validate
    np.random.seed(42)  # For reproducibility
    
    # Calculate metrics for each politician in each quarter
    all_politician_results = []
    for (quarter, rep, party), group in df.groupby(['Quarter', 'Representative', 'PartyName']):
        # Calculate the core metrics
        portfolio_value = group['TradeValueUpper'].sum()
        purchase_count = len(group[group['StandardizedTransaction'] == 'Purchase'])
        sale_count = len(group[group['StandardizedTransaction'] == 'Sale'])
        trade_count = len(group)
        unique_stocks = group['Ticker'].nunique()
        
        # Calculate weighted return with full precision
        raw_return = weighted_average(group)
        
        # For validation, manually check the calculation for a sample of politicians
        should_validate = np.random.random() < (VALIDATION_SAMPLE_SIZE / len(df.groupby(['Quarter', 'Representative'])))
        
        if should_validate or 'Mullin' in rep:
            print(f"\nValidating calculation for {rep} in {quarter}:")
            weights = group['TradeValueUpper'].values
            values = group['AdjustedPriceChange'].values
            valid_mask = ~np.isnan(values)
            valid_weights = weights[valid_mask]
            valid_values = values[valid_mask]
            
            if len(valid_weights) > 0:
                # Calculate manually to validate
                manual_weighted_sum = np.sum(valid_weights * valid_values)
                manual_total_weight = np.sum(valid_weights)
                manual_result = manual_weighted_sum / manual_total_weight
                
                # Compare with numpy's implementation
                numpy_result = np.average(valid_values, weights=valid_weights)
                
                # Print detailed calculation steps
                print(f"  Portfolio value: ${portfolio_value:,.2f}")
                print(f"  Trades: {trade_count} ({purchase_count} purchases, {sale_count} sales)")
                
                # Show the full calculation for a few trades as an example
                sample_size = min(3, len(valid_weights))
                print(f"  Sample calculation for {sample_size} trades:")
                for i in range(sample_size):
                    print(f"    Trade {i+1}: ${valid_weights[i]:,.2f} × {valid_values[i]:.6f}% = ${valid_weights[i] * valid_values[i] / 100:,.2f}")
                
                print(f"  Manual weighted sum: {manual_weighted_sum:.6f}")
                print(f"  Manual total weight: {manual_total_weight:.2f}")
                print(f"  Manual result: {manual_result:.6f}%")
                print(f"  NumPy result: {numpy_result:.6f}%")
                print(f"  Difference: {abs(manual_result - numpy_result):.10f}% (should be near zero)")
            else:
                print("  No valid trades with price data")
        
        # Store the result in our list
        all_politician_results.append({
            'Quarter': quarter,
            'Representative': rep,
            'PartyName': party,
            'PortfolioValue': portfolio_value,
            'WeightedReturn': raw_return,  # Full precision
            'PurchaseCount': purchase_count,
            'SaleCount': sale_count,
            'TradeCount': trade_count,
            'UniqueStocks': unique_stocks
        })
    
    # Convert to DataFrame
    politician_performance = pd.DataFrame(all_politician_results)
    
    # Add formatted version with consistent decimal places for display
    politician_performance['WeightedReturnFormatted'] = politician_performance['WeightedReturn'].apply(
        lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
    )
    
    # Add SPY reference performance for comparison
    politician_performance['SPYReturn'] = politician_performance['Quarter'].map(SPY_CHANGES)
    
    # Calculate excess return (how much the politician beat or trailed the market)
    politician_performance['ExcessReturn'] = politician_performance['WeightedReturn'] - politician_performance['SPYReturn']
    
    # Print summary statistics to verify overall calculation
    print("\nSummary statistics for weighted returns:")
    print(f"  Mean: {politician_performance['WeightedReturn'].mean():.4f}%")
    print(f"  Median: {politician_performance['WeightedReturn'].median():.4f}%")
    print(f"  Min: {politician_performance['WeightedReturn'].min():.4f}%")
    print(f"  Max: {politician_performance['WeightedReturn'].max():.4f}%")
    
    # Print verification of key politicians across different return ranges
    ranges = [
        (-float('inf'), -20),
        (-20, -10),
        (-10, 0),
        (0, 10),
        (10, 20),
        (20, float('inf'))
    ]
    
    print("\nVerification of politicians in different return ranges:")
    for low, high in ranges:
        filtered = politician_performance[
            (politician_performance['WeightedReturn'] >= low) & 
            (politician_performance['WeightedReturn'] < high)
        ]
        if len(filtered) > 0:
            print(f"\n  Range {low}% to {high}%: {len(filtered)} politicians")
            # Show a few examples from each range
            sample = filtered.sample(min(3, len(filtered)))
            for _, row in sample.iterrows():
                print(f"    {row['Representative']} ({row['PartyName']}): {row['WeightedReturn']:.4f}% (raw) → {row['WeightedReturnFormatted']}% (formatted)")
    
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
    
    # Add value labels on the right side of bars
    for i, (value, party, formatted_value) in enumerate(zip(
        top_performers['WeightedReturn'], 
        top_performers['PartyName'],
        top_performers['WeightedReturnFormatted']
    )):
        color = COLORS['text']  # Default text color
        
        # Place labels on the right side of the graph instead of at bar end
        # This ensures all labels are visible regardless of bar length
        bar_end_x = value
        label_x = max(bar_end_x + 1, 2)  # Ensure label is at least at x=2 position
        
        # Use the formatted value with consistent decimal places
        plt.text(label_x, i, f"{formatted_value}%", va='center', color=color)
        
        # Add party indicator next to politician name
        party_initial = 'D' if party == 'Democrats' else 'R'
        plt.text(-5, i, f"({party_initial})", va='center', ha='right', color=color, fontsize=10)
    
    # Add portfolio value annotation
    for i, (value, trades) in enumerate(zip(top_performers['PortfolioValue'], top_performers['TradeCount'])):
        plt.text(top_performers['WeightedReturn'].max() + 5, i, 
                 f"${value:,.0f} ({trades} trades)", va='center', color=COLORS['text'], fontsize=10)
    
    # Add extra space below the bottom-most bar for the SPY label
    plt.ylim(-2, len(top_performers))
    
    # Add a source watermark (moved to bottom margin)
    plt.figtext(0.5, 0.005, 'Data source: Quiver Quantitative', 
                fontsize=8, ha='center', color=COLORS['text'])
    
    # Add grid lines for readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust x-axis limits to include all labels
    right_margin = 25  # Add extra margin to fit all labels
    plt.xlim(left=min(-5, spy_return - 10), 
             right=max(top_performers['WeightedReturn'].max() + right_margin, spy_return + 10))
    
    # Save the figure with adjusted layout to make room for subtitle
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_path = os.path.join(output_dir, f'politician_performance_{quarter}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved politician performance visualization to: {output_path}")
    
    return output_path

def save_raw_data(df, output_dir):
    """Save the filtered data to CSV for reference."""
    # Add a more descriptive column name for clarity
    df = df.copy()
    df.rename(columns={'TradeValueUpper': 'TradeAmount_UpperRange_$'}, inplace=True)
    
    # Rearrange columns to put important ones first
    columns_order = [
        'Quarter', 'Representative', 'PartyName', 'Party',  
        'Ticker', 'Transaction', 'StandardizedTransaction',
        'TradeAmount_UpperRange_$', 'Range', 'Date', 'TransactionDate',
        'PriceChange', 'AdjustedPriceChange', 'SPYChange'
    ]
    
    # Only include columns that exist in the dataframe
    existing_columns = [col for col in columns_order if col in df.columns]
    
    # Add any remaining columns at the end
    remaining_columns = [col for col in df.columns if col not in existing_columns]
    
    # Set the final column order
    final_columns = existing_columns + remaining_columns
    
    # Save to CSV with the specified column order
    output_path = os.path.join(output_dir, 'politician_performance_raw_data.csv')
    df[final_columns].to_csv(output_path, index=False)
    print(f"Saved filtered raw data to: {output_path}")
    
    return output_path

def main():
    """Main function to analyze individual politician performance and create visualizations."""
    print("Starting Individual Politician Performance analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    # Setup output directory
    output_dir = setup_directories()
    
    try:
        # Check if API key is set
        if not QUIVER_API_KEY:
            print("Error: API key not set. Please create a .env file with your API_KEY=your_api_key_here")
            print("You can sign up for a Quiver Quantitative API key at https://www.quiverquant.com/")
            return
        
        # Get trading data
        trading_data = get_congress_trading_data()
        
        if trading_data is not None:
            # Preprocess data
            df_processed = preprocess_data(trading_data)
            
            if df_processed is not None and not df_processed.empty:
                # Save raw filtered data
                save_raw_data(df_processed, output_dir)
                
                # Calculate politician performance metrics
                politician_performance = calculate_politician_performance(df_processed)
                
                # Save performance data
                performance_csv_path = os.path.join(output_dir, 'politician_performance_data.csv')
                politician_performance.to_csv(performance_csv_path, index=False)
                print(f"Saved politician performance data to: {performance_csv_path}")
                
                # Create visualizations for each quarter - include ALL politicians
                for quarter in ['2024-Q4', '2025-Q1']:
                    plot_politician_performance(politician_performance, output_dir, quarter)
                
                print("Individual Politician Performance analysis complete. All outputs saved to the Politician_Performance directory.")
            else:
                print("Error: No valid data available for analysis after filtering.")
        else:
            print("Failed to fetch trading data. Please check your API key and try again.")
        
    except Exception as e:
        print(f"Error during Politician Performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 