"""
Party Performance Analysis

This script analyzes congressional trading activity by political party (Democrats vs Republicans),
comparing performance for two time periods:
- 2024 Q4 (October 1, 2024 - December 31, 2024)
- 2025 Q1 (January 1, 2025 - March 21, 2025)

The analysis calculates:
- Total portfolio value changes by party
- Comparison to market (SPY) performance
- Weighted percent change visualization

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
    '2025-Q1': (SPY_PRICES['q1_end'] - SPY_PRICES['q1_start']) / SPY_PRICES['q1_start'] * 100,
    'Combined': (SPY_PRICES['q1_end'] - SPY_PRICES['q4_start']) / SPY_PRICES['q4_start'] * 100
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

def setup_directories():
    """Ensure the Party_Performance directory exists."""
    party_dir = os.path.dirname(__file__)
    os.makedirs(party_dir, exist_ok=True)
    return party_dir

def setup_plot_style():
    """Setup global plotting style for a dark theme with political colors."""
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

def calculate_party_performance(df):
    """
    Calculate performance metrics by party, accounting for trade direction.
    
    Steps:
    1. Adjust price change based on transaction type (Purchase vs Sale)
    2. Calculate weighted returns for each party in each quarter
    3. Compare to SPY performance
    """
    print("Calculating party performance metrics...")
    
    # STEP 1: Create adjusted price change accounting for transaction type
    # For purchases: positive price change = profit
    # For sales: negative price change = profit (reversed sign)
    df['AdjustedPriceChange'] = df.apply(
        lambda row: row['PriceChange'] if row['StandardizedTransaction'] == 'Purchase' else -row['PriceChange'], 
        axis=1
    )
    
    # Function to calculate weighted average return for a group of trades
    def weighted_average(group):
        # STEP 2a: Extract trade values (weights) and adjusted returns (values)
        # TradeValueUpper = the UPPER/LARGER value from the Range column (e.g. for "$1,001-$15,000" it's $15,000)
        weights = group['TradeValueUpper']  # Dollar amount of each trade (upper range value)
        values = group['AdjustedPriceChange']  # Direction-adjusted price change percentage
        
        # Filter out NaN values from both arrays
        mask = ~values.isna()
        weights, values = weights[mask], values[mask]
        
        # If no valid data, return NaN
        if len(weights) == 0 or weights.sum() == 0:
            return np.nan
        
        # STEP 2b: Calculate weighted average using numpy's average function
        # Formula: sum(weight_i * value_i) / sum(weights)
        # Each trade contributes proportionally to its dollar amount
        return np.average(values, weights=weights)
    
    # STEP 3: Calculate metrics for each party in each quarter
    party_performance = df.groupby(['Quarter', 'PartyName']).apply(
        lambda x: pd.Series({
            # Sum of all trade dollar amounts for this party in this quarter
            'PortfolioValue': x['TradeValueUpper'].sum(),
            
            # Weighted average return calculation (calls the function above)
            # This accounts for both transaction direction and trade size
            'WeightedReturn': weighted_average(x),
            
            # Count metrics for context
            'PurchaseCount': len(x[x['StandardizedTransaction'] == 'Purchase']),
            'SaleCount': len(x[x['StandardizedTransaction'] == 'Sale']),
            'OtherCount': len(x[~x['StandardizedTransaction'].isin(['Purchase', 'Sale'])]),
            'TradeCount': len(x),
            'UniquePoliticians': x['Representative'].nunique(),
            'UniqueStocks': x['Ticker'].nunique()
        })
    ).reset_index()
    
    # STEP 4: Add SPY reference performance for comparison
    party_performance['SPYReturn'] = party_performance['Quarter'].map(SPY_CHANGES)
    
    # STEP 5: Calculate excess return (how much the party portfolio beat or trailed the market)
    party_performance['ExcessReturn'] = party_performance['WeightedReturn'] - party_performance['SPYReturn']
    
    # STEP 6: Calculate relative performance metrics
    party_performance['MarketOutperformance'] = np.where(
        party_performance['WeightedReturn'] > party_performance['SPYReturn'],
        'Outperformed',
        'Underperformed'
    )
    
    # Print detailed summary for verification
    print("\nParty Performance Summary:")
    for _, row in party_performance.iterrows():
        print(f"{row['PartyName']} in {row['Quarter']}:")
        print(f"  Portfolio Value: ${row['PortfolioValue']:,.2f}")
        print(f"  Weighted Return: {row['WeightedReturn']:.2f}%")
        print(f"  SPY Return: {row['SPYReturn']:.2f}%")
        print(f"  Excess Return: {row['ExcessReturn']:.2f}%")
        print(f"  Trade Counts: {row['PurchaseCount']} purchases, {row['SaleCount']} sales")
        print(f"  {row['UniquePoliticians']} politicians, {row['TradeCount']} trades, {row['UniqueStocks']} unique stocks")
        print()
    
    return party_performance

def plot_party_performance(performance_data, output_dir):
    """
    Create separate horizontal bar charts for each quarter showing performance by party.
    """
    print("Creating party performance visualizations...")
    
    # Split the visualization into separate charts by quarter
    for quarter in ['2024-Q4', '2025-Q1']:
        print(f"Creating chart for {quarter}...")
        
        # Filter for the selected quarter
        quarter_data = performance_data[performance_data['Quarter'] == quarter].copy()
        
        # Get the SPY return for this quarter
        spy_return = SPY_CHANGES[quarter]
        
        # Create a dataframe for plotting with consistent order
        # Democrats on top, Republicans in middle, SPY at bottom
        plot_data = pd.DataFrame([
            {'Party': 'Democrats', 'Return': quarter_data[quarter_data['PartyName'] == 'Democrats']['WeightedReturn'].values[0]},
            {'Party': 'Republicans', 'Return': quarter_data[quarter_data['PartyName'] == 'Republicans']['WeightedReturn'].values[0]},
            {'Party': 'SPY', 'Return': spy_return}
        ])
        
        # Create the plot - horizontal bar chart
        plt.figure(figsize=(12, 6))
        
        # Create positions for bars (top to bottom)
        positions = np.arange(len(plot_data))
        
        # Create color map based on party
        colors = [
            COLORS['democrat'],  # Democrats
            COLORS['republican'],  # Republicans
            COLORS['spy']   # SPY
        ]
        
        # Plot horizontal bars
        bars = plt.barh(positions, plot_data['Return'], height=0.6, color=colors)
        
        # Add labels and title
        plt.title(f'Weighted Percent Change of Party Portfolios - {quarter}', fontsize=20, pad=20)
        
        # Move the subtitle below the x-axis label to avoid overlap with title
        plt.xlabel('Percent Change (%)', fontsize=14, labelpad=10)
        plt.figtext(0.5, 0.02, 
                    'Each bar shows the change in stock portfolio value for the specified period', 
                    fontsize=12, ha='center', color=COLORS['text'])
        
        # Format y-axis with party names
        plt.yticks(positions, plot_data['Party'])
        plt.ylabel('Party', fontsize=14)
        
        # Add value labels on the right side of bars
        for i, value in enumerate(plot_data['Return']):
            # Place labels on the right side of the graph instead of at bar end
            bar_end_x = value
            label_x = max(bar_end_x + 1, 2)  # Ensure label is at least at x=2 position
            
            # Add value label with consistent formatting
            plt.text(label_x, i, f"{value:.1f}%", va='center', color=COLORS['text'])
        
        # Add extra space below the bottom-most bar for the SPY label
        plt.ylim(-1, len(plot_data))
        
        # Add SPY value label on the x-axis (below the bars)
        plt.text(spy_return, -0.5, f"{spy_return:.1f}%", 
                 ha='center', va='center', color=COLORS['spy'], fontweight='bold',
                 bbox=dict(facecolor=COLORS['background'], edgecolor=COLORS['spy'], boxstyle='round,pad=0.5'))
        
        # Add a source watermark (moved to bottom margin)
        plt.figtext(0.5, 0.005, 'Data source: Quiver Quantitative', 
                    fontsize=8, ha='center', color=COLORS['text'])
        
        # Add grid lines for readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust x-axis limits to include all labels
        right_margin = 5  # Add extra margin to fit all labels
        plt.xlim(left=min(-1, spy_return - 5), 
                 right=max(plot_data['Return'].max() + right_margin, spy_return + 5))
        
        # Save the figure with adjusted layout to make room for subtitle
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        output_path = os.path.join(output_dir, f'party_performance_{quarter}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
        print(f"Saved party performance visualization for {quarter} to: {output_path}")
        plt.close()
    
    print("Party performance visualizations complete.")

def save_raw_data(df, output_dir):
    """Save the filtered data to CSV for reference, with trade upper amount prominently included."""
    # Ensure TradeValueUpper column is properly formatted as currency
    if 'TradeValueUpper' in df.columns:
        print(f"Including TradeValueUpper column in raw data output (range: ${df['TradeValueUpper'].min():,.2f} to ${df['TradeValueUpper'].max():,.2f})")
    else:
        print("Warning: TradeValueUpper column not found in data")
        
    # Add a more descriptive column name for clarity
    df = df.copy()
    df.rename(columns={'TradeValueUpper': 'TradeAmount_UpperRange_$'}, inplace=True)
    
    # Rearrange columns to put important ones first
    columns_order = [
        'Quarter', 'PartyName', 'Party', 'Representative', 
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
    output_path = os.path.join(output_dir, 'party_performance_raw_data.csv')
    df[final_columns].to_csv(output_path, index=False)
    print(f"Saved filtered raw data to: {output_path}")
    
    # Print verification of the columns included
    print(f"Raw data CSV includes {len(df)} rows with {len(final_columns)} columns")
    print(f"First 5 columns: {', '.join(final_columns[:5])}")
    
    return output_path

def analyze_transaction_types(df, output_dir):
    """
    Create a detailed analysis of transaction types (Purchase vs. Sale) by party.
    """
    # Group by Party, Quarter, and Transaction
    tx_breakdown = df.groupby(['PartyName', 'Quarter', 'StandardizedTransaction']).agg({
        'TradeValueUpper': 'sum',
        'Representative': 'nunique',
        'PriceChange': ['mean', 'median']
    }).reset_index()
    
    # Save to CSV for review
    tx_breakdown.to_csv(os.path.join(output_dir, 'transaction_type_breakdown.csv'))
    
    # Create visualization showing Purchase vs. Sale distribution by party
    # ... (visualization code) ...

def main():
    """Main function to analyze party performance and create visualizations."""
    print("Starting Party Performance analysis...")
    
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
                
                # Calculate party performance metrics
                party_performance = calculate_party_performance(df_processed)
                
                # Create visualizations
                plot_party_performance(party_performance, output_dir)
                
                # Analyze transaction types
                analyze_transaction_types(df_processed, output_dir)
                
                print("Party Performance analysis complete. All outputs saved to the Party_Performance directory.")
            else:
                print("Error: No valid data available for analysis after filtering.")
        else:
            print("Failed to fetch trading data. Please check your API key and try again.")
        
    except Exception as e:
        print(f"Error during Party Performance analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 