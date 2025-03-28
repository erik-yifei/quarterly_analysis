"""
Stock Holdings Analysis By Party and Transaction Type

This script analyzes stock holdings and sales by party, displaying:
- What stocks Democrats are buying vs. selling
- What stocks Republicans are buying vs. selling
- Comparing across two time periods:
  - 2024 Q4 (October 1, 2024 - December 31, 2024)
  - 2025 Q1 (January 1, 2025 - March 21, 2025)

The analysis is displayed as pie charts showing the distribution
of stock holdings/sales by ticker for each party and transaction type.
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

# Define custom color scheme - dark theme with blue/red political colors
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'democrat': '#3C99DC',      # Democrat blue
    'republican': '#E91D0E',    # Republican red
    # Colorful palette for pie chart slices
    'pie_palette': [
        '#4285F4',  # Google Blue
        '#EA4335',  # Google Red
        '#FBBC05',  # Google Yellow
        '#34A853',  # Google Green
        '#FF6D01',  # Orange
        '#46BDC6',  # Cyan
        '#7E57C2',  # Purple
        '#EC407A',  # Pink
        '#5C6BC0',  # Indigo
        '#26A69A',  # Teal
        '#FFA726',  # Amber
        '#78909C',  # Blue Grey
        '#AB47BC',  # Purple
        '#66BB6A',  # Light Green
        '#42A5F5',  # Light Blue
        '#FFA000',  # Amber
        '#EC407A',  # Pink
        '#5C6BC0',  # Indigo
    ]
}

def setup_directories():
    """Ensure the Stock_Analysis directory exists."""
    stock_dir = os.path.dirname(__file__)
    os.makedirs(stock_dir, exist_ok=True)
    return stock_dir

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
        client = quiverquant.quiver(QUIVER_API_KEY)
        
        # Get congressional trading data (single method, not separate house/senate)
        print("Fetching Congressional trading data...")
        congress_data = client.congress_trading()
        
        print(f"Retrieved {len(congress_data)} trading records")
        return congress_data
        
    except Exception as e:
        print(f"Error fetching congressional trading data: {str(e)}")
        return None

def parse_range_values(range_str):
    """
    Parse the 'Range' column to extract the upper dollar amount.
    Example: "$1,001 - $15,000" -> 15000
    """
    if pd.isna(range_str):
        return 0
        
    # Extract numbers from the range string
    numbers = re.findall(r'[$]([0-9,]+)', str(range_str))
    if not numbers or len(numbers) < 2:
        return 0
    
    # Take the higher number (second value in the range)
    return float(numbers[1].replace(',', ''))

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

def analyze_stock_holdings(df):
    """
    Analyze stock holdings by party and transaction type.
    Creates summaries of what stocks each party is buying vs. selling.
    """
    print("Analyzing stock holdings by party and transaction type...")
    
    # Group by Quarter, Party, Transaction Type, and Ticker
    # Sum the TradeValueUpper to get total value for each stock
    stock_holdings = df.groupby(['Quarter', 'PartyName', 'StandardizedTransaction', 'Ticker']).agg({
        'TradeValueUpper': 'sum',
        'Representative': 'nunique',
        'Transaction': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    stock_holdings = stock_holdings.rename(columns={
        'TradeValueUpper': 'TotalValue',
        'Representative': 'UniquePoliticians',
        'Transaction': 'TransactionCount'
    })
    
    # Calculate the total by Quarter, Party, and Transaction Type
    # This will be used to calculate percentage of each stock
    totals = stock_holdings.groupby(['Quarter', 'PartyName', 'StandardizedTransaction'])['TotalValue'].sum().reset_index()
    totals = totals.rename(columns={'TotalValue': 'CategoryTotal'})
    
    # Merge the totals back to get percentages
    stock_holdings = pd.merge(
        stock_holdings, 
        totals, 
        on=['Quarter', 'PartyName', 'StandardizedTransaction']
    )
    
    # Calculate percentage of total
    stock_holdings['Percentage'] = stock_holdings['TotalValue'] / stock_holdings['CategoryTotal'] * 100
    
    # Sort by value within each group
    stock_holdings = stock_holdings.sort_values(
        ['Quarter', 'PartyName', 'StandardizedTransaction', 'TotalValue'], 
        ascending=[True, True, True, False]
    )
    
    # Print summary stats
    print("\nTop holdings by party and transaction type:")
    for quarter in stock_holdings['Quarter'].unique():
        print(f"\n{quarter}:")
        for party in stock_holdings['PartyName'].unique():
            print(f"  {party}:")
            for tx_type in stock_holdings['StandardizedTransaction'].unique():
                subset = stock_holdings[
                    (stock_holdings['Quarter'] == quarter) & 
                    (stock_holdings['PartyName'] == party) & 
                    (stock_holdings['StandardizedTransaction'] == tx_type)
                ]
                if len(subset) > 0:
                    total = subset['CategoryTotal'].iloc[0]
                    print(f"    {tx_type} (Total: ${total:,.2f}):")
                    for _, row in subset.head(5).iterrows():
                        print(f"      {row['Ticker']}: ${row['TotalValue']:,.2f} ({row['Percentage']:.1f}% of total)")
    
    # Save to CSV for reference
    output_path = os.path.join(os.path.dirname(__file__), 'stock_holdings_by_party.csv')
    stock_holdings.to_csv(output_path, index=False)
    print(f"Saved stock holdings data to: {output_path}")
    
    return stock_holdings

def plot_stock_pie_charts(stock_data, output_dir):
    """
    Create pie charts showing stock distribution for each party and transaction type.
    
    Parameters:
        stock_data: DataFrame with stock holdings analysis
        output_dir: Directory to save the visualizations
    """
    print("Creating stock holdings pie charts...")
    
    # Get unique quarters, parties, and transaction types
    quarters = stock_data['Quarter'].unique()
    parties = stock_data['PartyName'].unique()
    tx_types = stock_data['StandardizedTransaction'].unique()
    
    # For each combination, create a pie chart
    for quarter in quarters:
        for party in parties:
            for tx_type in tx_types:
                # Filter data for this combination
                subset = stock_data[
                    (stock_data['Quarter'] == quarter) & 
                    (stock_data['PartyName'] == party) & 
                    (stock_data['StandardizedTransaction'] == tx_type)
                ]
                
                if len(subset) == 0:
                    print(f"No data for {party} {tx_type} in {quarter}")
                    continue
                
                # For clarity, only show top 10 stocks explicitly, group the rest as "Other"
                top_10 = subset.head(10).copy()
                others = subset.iloc[10:].copy() if len(subset) > 10 else None
                
                if others is not None and len(others) > 0:
                    other_sum = others['TotalValue'].sum()
                    other_percent = others['TotalValue'].sum() / subset['CategoryTotal'].iloc[0] * 100
                    other_row = pd.DataFrame({
                        'Quarter': [quarter],
                        'PartyName': [party],
                        'StandardizedTransaction': [tx_type],
                        'Ticker': ['Other'],
                        'TotalValue': [other_sum],
                        'UniquePoliticians': [others['UniquePoliticians'].sum()],
                        'TransactionCount': [others['TransactionCount'].sum()],
                        'CategoryTotal': [subset['CategoryTotal'].iloc[0]],
                        'Percentage': [other_percent]
                    })
                    plot_data = pd.concat([top_10, other_row], ignore_index=True)
                else:
                    plot_data = top_10
                
                # Create the pie chart
                plt.figure(figsize=(12, 9))
                
                # Use a nice color palette
                colors = COLORS['pie_palette'][:len(plot_data)]
                
                # Create pie chart
                wedges, texts, autotexts = plt.pie(
                    plot_data['TotalValue'],
                    labels=None,  # No labels on the pie itself
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                    textprops={'color': COLORS['text']}
                )
                
                # Enhance the appearance of percentage labels
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(9)
                    autotext.set_fontweight('bold')
                
                # Add a title with total amount
                total_amount = subset['CategoryTotal'].iloc[0]
                plt.title(
                    f"{party} {tx_type}s by Stock - {quarter}\nTotal: ${total_amount:,.0f}",
                    fontsize=18, 
                    pad=20,
                    color=COLORS['text']
                )
                
                # Create a legend with ticker and value
                ticker_labels = [
                    f"{row['Ticker']} (${row['TotalValue']:,.0f})"
                    for _, row in plot_data.iterrows()
                ]
                plt.legend(
                    wedges, 
                    ticker_labels,
                    title="Stock (Total Value)",
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1),
                    fontsize=10,
                )
                
                # Add a source watermark
                plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', 
                            fontsize=8, ha='center', color=COLORS['text'])
                
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
                # Add party color to the background
                party_color = COLORS['democrat'] if party == 'Democrats' else COLORS['republican']
                circle = plt.Circle((0, 0), 0.6, color=party_color, alpha=0.1)
                plt.gcf().gca().add_artist(circle)
                
                # Save the figure
                plt.tight_layout()
                output_path = os.path.join(
                    output_dir, 
                    f'stock_{party.lower()}_{tx_type.lower()}_{quarter}.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
                print(f"Saved pie chart to: {output_path}")
                plt.close()
    
    print("All stock pie charts created successfully.")

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
    output_path = os.path.join(output_dir, 'all_trades_raw_data.csv')
    df[final_columns].to_csv(output_path, index=False)
    print(f"Saved complete raw trade data to: {output_path}")
    
    # Save individual CSVs by party and transaction type for easier analysis
    for party in df['PartyName'].unique():
        for tx_type in df['StandardizedTransaction'].unique():
            subset = df[
                (df['PartyName'] == party) &
                (df['StandardizedTransaction'] == tx_type)
            ]
            if len(subset) > 0:
                party_code = 'D' if party == 'Democrats' else 'R'
                tx_code = 'P' if tx_type == 'Purchase' else 'S'
                subset_path = os.path.join(output_dir, f'trades_{party_code}_{tx_code}.csv')
                subset[final_columns].to_csv(subset_path, index=False)
                print(f"Saved {party} {tx_type}s data: {len(subset)} records to {subset_path}")
    
    return output_path

def main():
    """Main function to analyze stock holdings and create visualizations."""
    print("Starting Stock Holdings analysis...")
    
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
                # Save raw filtered data for Excel analysis
                save_raw_data(df_processed, output_dir)
                
                # Analyze stock holdings
                stock_holdings = analyze_stock_holdings(df_processed)
                
                # Create visualizations
                plot_stock_pie_charts(stock_holdings, output_dir)
                
                print("Stock Holdings analysis complete. All outputs saved to the Stock_Analysis directory.")
            else:
                print("Error: No valid data available for analysis after filtering.")
        else:
            print("Failed to fetch trading data. Please check your API key and try again.")
        
    except Exception as e:
        print(f"Error during Stock Holdings analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 