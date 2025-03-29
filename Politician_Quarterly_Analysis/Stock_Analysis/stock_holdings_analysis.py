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
import math
from matplotlib.colors import LinearSegmentedColormap

# Add INPUT_DIR definition
INPUT_DIR = r"C:\Users\ErikWang\Documents\new_poli_analysis\Politician_Quarterly_Analysis\Input"

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
    """Preprocess the data with the same logic as party performance"""
    print("Preprocessing data...")
    
    # Convert dates to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Clean up Transaction types
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
    
    # Add cleaned transaction type column
    df['TransactionType'] = df['Transaction'].apply(clean_transaction_type)
    
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
    
    # Filter for the two quarters of interest
    q4_2024_start = pd.Timestamp('2024-10-01')
    q4_2024_end = pd.Timestamp('2024-12-31')
    q1_2025_start = pd.Timestamp('2025-01-01')
    q1_2025_end = pd.Timestamp('2025-03-21')
    
    conditions = [
        (df['TransactionDate'] >= q4_2024_start) & (df['TransactionDate'] <= q4_2024_end),
        (df['TransactionDate'] >= q1_2025_start) & (df['TransactionDate'] <= q1_2025_end)
    ]
    choices = ['2024-Q4', '2025-Q1']
    df['Quarter'] = np.select(conditions, choices, default='Other')
    
    # Filter for the quarters we want
    df = df[df['Quarter'].isin(['2024-Q4', '2025-Q1'])]
    
    print("\nData Summary:")
    print(f"Total trades (excluding exchanges): {len(df)}")
    print(f"Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
    print(f"\nTrades by transaction type:")
    print(df['TransactionType'].value_counts())
    print(f"\nTrades by party:")
    print(df['PartyName'].value_counts())
    print(f"\nTotal trade amount: ${df['TradeAmount'].sum():,.2f}")
    
    return df

def analyze_stock_holdings(df):
    """Analyze stock holdings by party and transaction type."""
    print("Analyzing stock holdings by party and transaction type...")
    
    # Group by Quarter, Party, Transaction Type, and Ticker
    stock_holdings = df.groupby(['Quarter', 'PartyName', 'TransactionType', 'Ticker']).agg({
        'TradeAmount': 'sum',  # Use TradeAmount instead of TradeValueUpper
        'Representative': 'nunique',
        'Transaction': 'count'
    }).reset_index()
    
    # Rename columns for clarity
    stock_holdings = stock_holdings.rename(columns={
        'TradeAmount': 'TotalValue',
        'Representative': 'UniquePoliticians',
        'Transaction': 'TransactionCount'
    })
    
    # Calculate the total by Quarter, Party, and Transaction Type
    totals = stock_holdings.groupby(['Quarter', 'PartyName', 'TransactionType'])['TotalValue'].sum().reset_index()
    totals = totals.rename(columns={'TotalValue': 'CategoryTotal'})
    
    # Merge the totals back to get percentages
    stock_holdings = pd.merge(
        stock_holdings, 
        totals, 
        on=['Quarter', 'PartyName', 'TransactionType']
    )
    
    # Calculate percentage of total
    stock_holdings['Percentage'] = stock_holdings['TotalValue'] / stock_holdings['CategoryTotal'] * 100
    
    # Sort by value within each group
    stock_holdings = stock_holdings.sort_values(
        ['Quarter', 'PartyName', 'TransactionType', 'TotalValue'], 
        ascending=[True, True, True, False]
    )
    
    # Print summary stats
    print("\nTop holdings by party and transaction type:")
    for quarter in stock_holdings['Quarter'].unique():
        print(f"\n{quarter}:")
        for party in stock_holdings['PartyName'].unique():
            print(f"  {party}:")
            for tx_type in stock_holdings['TransactionType'].unique():
                subset = stock_holdings[
                    (stock_holdings['Quarter'] == quarter) & 
                    (stock_holdings['PartyName'] == party) & 
                    (stock_holdings['TransactionType'] == tx_type)
                ]
                if len(subset) > 0:
                    total = subset['CategoryTotal'].iloc[0]
                    print(f"    {tx_type} (Total: ${total:,.2f}):")
                    for _, row in subset.head(5).iterrows():
                        print(f"      {row['Ticker']}: ${row['TotalValue']:,.2f} ({row['Percentage']:.1f}% of total)")
    
    return stock_holdings

def plot_stock_pie_charts(stock_data, output_dir):
    """Create side-by-side pie charts showing stock distribution for purchases and sales."""
    print("Creating stock holdings pie charts...")
    
    # Get unique quarters and parties
    quarters = stock_data['Quarter'].unique()
    parties = stock_data['PartyName'].unique()
    
    # For each quarter and party combination, create side-by-side pie charts
    for quarter in quarters:
        for party in parties:
            # Create a figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            
            # Process each transaction type (purchase and sale)
            for tx_type, ax in zip(['purchase', 'sale'], [ax1, ax2]):
                # Filter data for this combination
                subset = stock_data[
                    (stock_data['Quarter'] == quarter) & 
                    (stock_data['PartyName'] == party) & 
                    (stock_data['TransactionType'] == tx_type)
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
                        'TransactionType': [tx_type],
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
                
                # Use a nice color palette
                colors = COLORS['pie_palette'][:len(plot_data)]
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(
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
                ax.set_title(
                    f"{tx_type.capitalize()}s\nTotal: ${total_amount:,.0f}",
                    fontsize=16, 
                    pad=20,
                    color=COLORS['text']
                )
                
                # Create a legend with ticker and value
                ticker_labels = [
                    f"{row['Ticker']} (${row['TotalValue']:,.0f})"
                    for _, row in plot_data.iterrows()
                ]
                ax.legend(
                    wedges, 
                    ticker_labels,
                    title="Stock (Total Value)",
                    loc="center left",
                    bbox_to_anchor=(1.1, 0.5),
                    fontsize=10,
                )
                
                # Equal aspect ratio ensures that pie is drawn as a circle
                ax.axis('equal')
                
                # Add party color to the background
                party_color = COLORS['democrat'] if party == 'Democrats' else COLORS['republican']
                circle = plt.Circle((0, 0), 0.6, color=party_color, alpha=0.1, transform=ax.transData)
                ax.add_artist(circle)
            
            # Add an overall title for the figure
            plt.suptitle(
                f"{party} Stock Trading Activity - {quarter}",
                fontsize=22, 
                y=1.02,
                color=COLORS['text']
            )
            
            # Add a source watermark
            plt.figtext(0.5, 0.02, 'Data source: House Stock Watcher', 
                        fontsize=8, ha='center', color=COLORS['text'])
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_path = os.path.join(
                output_dir, 
                f'stock_{party.lower()}_combined_{quarter}.png'
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
            print(f"Saved combined pie charts to: {output_path}")
            plt.close()
    
    print("All stock pie charts created successfully.")

def save_raw_data(df, output_dir):
    """Save the filtered data to CSV for reference."""
    print("\nPreparing raw data for CSV output...")
    
    # Create a copy to avoid modifying the original dataframe
    df_output = df.copy()
    
    # Extract upper range value from Range column
    def extract_upper_range(range_str):
        try:
            # Extract the second number (upper limit) from strings like "$1,001 - $15,000"
            upper_amount = range_str.split('-')[1].strip()
            # Remove '$' and ',' then convert to float
            return float(upper_amount.replace('$', '').replace(',', ''))
        except:
            return 0.0
    
    # Add upper range column
    df_output['TradeAmount_UpperRange'] = df_output['Range'].apply(extract_upper_range)
    
    # Format as currency string
    df_output['TradeAmount_UpperRange_$'] = df_output['TradeAmount_UpperRange'].apply(
        lambda x: f"${x:,.2f}"
    )
    
    # Select and reorder columns
    columns_to_keep = [
        'Quarter',
        'PartyName', 
        'Representative',
        'Ticker',
        'TransactionType',
        'TradeAmount_UpperRange_$',
        'TradeAmount_UpperRange',  # Numeric version for calculations
        'PercentChange',
        'TransactionDate'
    ]
    
    # Only keep columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df_output.columns]
    
    # Save main dataset
    main_output = os.path.join(output_dir, 'cleaned_trading_data.csv')
    df_output[existing_columns].to_csv(main_output, index=False)
    print(f"Saved all trades to: {main_output}")
    
    # Save separate files by party and transaction type
    for party in df_output['PartyName'].unique():
        for tx_type in df_output['TransactionType'].unique():
            subset = df_output[
                (df_output['PartyName'] == party) &
                (df_output['TransactionType'] == tx_type)
            ]
            if len(subset) > 0:
                party_code = 'D' if party == 'Democrats' else 'R'
                tx_code = 'P' if tx_type == 'purchase' else 'S'
                subset_path = os.path.join(output_dir, f'trades_{party_code}_{tx_code}.csv')
                subset[existing_columns].to_csv(subset_path, index=False)
                print(f"Saved {party} {tx_type}s data: {len(subset)} records to {subset_path}")
    
    # Print summary of trade amounts
    print("\nTrade Amount Summary:")
    print(f"Total trade amount: ${df_output['TradeAmount_UpperRange'].sum():,.2f}")
    print(f"Average trade amount: ${df_output['TradeAmount_UpperRange'].mean():,.2f}")
    print(f"Median trade amount: ${df_output['TradeAmount_UpperRange'].median():,.2f}")
    print(f"Range: ${df_output['TradeAmount_UpperRange'].min():,.2f} to "
          f"${df_output['TradeAmount_UpperRange'].max():,.2f}")
    
    return main_output

def main():
    """Main function to analyze stock holdings and create visualizations."""
    print("Starting Stock Holdings analysis...")
    
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
                # Save raw filtered data for reference
                save_raw_data(df_processed, output_dir)
                
                # Analyze stock holdings
                stock_holdings = analyze_stock_holdings(df_processed)
                
                # Create visualizations
                plot_stock_pie_charts(stock_holdings, output_dir)
                
                print("Stock Holdings analysis complete. All outputs saved to the Stock_Analysis directory.")
            else:
                print("Error: No valid data available for analysis after filtering.")
        else:
            print("Error: Could not load data from final_results.csv")
        
    except Exception as e:
        print(f"Error during Stock Holdings analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 