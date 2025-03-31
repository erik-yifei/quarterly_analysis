"""
Selected Traders Analysis

Analyzes stock holdings and performance for specific traders:

2025 Q1:
- Thomas H. Kean, Jr.
- Tommy Tuberville
- Robert Bresnahan

2024 Q4:
- Morgan Mcgravey
- Nancy Pelosi
- Markwayne Mullin

Shows:
1. What stocks they traded
2. Performance of each asset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

# Define constants
INPUT_DIR = r"C:\Users\ErikWang\Documents\new_poli_analysis\Politician_Quarterly_Analysis\Input"

# Define custom color scheme - dark theme
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'democrat': '#3C99DC',      # Democrat blue
    'republican': '#E91D0E',    # Republican red
    'positive': '#00FF00',      # Green for positive returns
    'negative': '#FF0000',      # Red for negative returns
}

# Selected traders - only these specific ones
SELECTED_TRADERS = {
    '2025-Q1': [
        'Thomas H. Kean Jr',
        'Tommy Tuberville',
        'Rob Bresnahan'
    ],
    '2024-Q4': [
        'Morgan Mcgarvey',
        'Nancy Pelosi',
        'Markwayne Mullin'
    ]
}

def setup_directories():
    """Create output directory for Selected Traders Analysis"""
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_plot_style():
    """Setup global plotting style for a dark theme."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif'
    })

def parse_range_values(range_str):
    """Parse the Range column to extract the upper dollar amount."""
    if pd.isna(range_str):
        return 0
    
    numbers = re.findall(r'[$]([0-9,]+)', str(range_str))
    if not numbers or len(numbers) < 2:
        return 0
    
    try:
        return float(numbers[1].replace(',', ''))
    except (ValueError, IndexError):
        return 0

def load_and_filter_data():
    """Load data and filter for selected traders"""
    file_path = os.path.join(INPUT_DIR, 'final_results.csv')
    df = pd.read_csv(file_path)
    
    # Convert dates
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Filter for quarters
    q4_2024_start = pd.Timestamp('2024-10-01')
    q4_2024_end = pd.Timestamp('2024-12-31')
    q1_2025_start = pd.Timestamp('2025-01-01')
    q1_2025_end = pd.Timestamp('2025-03-28')
    
    conditions = [
        (df['TransactionDate'] >= q4_2024_start) & (df['TransactionDate'] <= q4_2024_end),
        (df['TransactionDate'] >= q1_2025_start) & (df['TransactionDate'] <= q1_2025_end)
    ]
    choices = ['2024-Q4', '2025-Q1']
    df['Quarter'] = np.select(conditions, choices, default='Other')
    
    # Filter for selected traders
    all_selected = [trader for traders in SELECTED_TRADERS.values() for trader in traders]
    df = df[df['Representative'].isin(all_selected)]
    
    # Calculate trade amounts
    df['TradeAmount'] = df['Range'].apply(parse_range_values)
    
    # Calculate returns based on Transaction type
    def calculate_return(row):
        if row['Transaction'] in ['Sale (Partial)', 'Sale (Full)']:
            return -1 * row['PercentChange']
        return row['PercentChange']
    
    df['AdjustedReturn'] = df.apply(calculate_return, axis=1)
    
    return df

def analyze_trader_performance(df):
    """Analyze performance for each trader and their assets"""
    def weighted_return_by_ticker(group):
        # Group trades by ticker and calculate weighted return for each ticker
        ticker_groups = group.groupby('Ticker')
        ticker_returns = {}
        
        for ticker, trades in ticker_groups:
            # Calculate weighted return for this ticker
            total_amount = trades['TradeAmount'].sum()
            weighted_return = (trades['PercentChange'] * trades['TradeAmount']).sum() / total_amount
            
            # For sales, flip the sign
            if all(t in ['Sale (Partial)', 'Sale (Full)'] for t in trades['Transaction']):
                weighted_return = -weighted_return
            
            ticker_returns[ticker] = {
                'AdjustedReturn': weighted_return,
                'TradeAmount': total_amount,
                'Transaction': trades['Transaction'].tolist(),
                'TransactionDate': trades['TransactionDate'].tolist(),
                'PercentChange': trades['PercentChange'].tolist()
            }
        
        return pd.DataFrame.from_dict(ticker_returns, orient='index').reset_index().rename(columns={'index': 'Ticker'})
    
    # Group by trader and quarter first
    trader_assets = []
    for (quarter, rep), group in df.groupby(['Quarter', 'Representative']):
        # Get weighted returns for each ticker
        ticker_data = weighted_return_by_ticker(group)
        ticker_data['Quarter'] = quarter
        ticker_data['Representative'] = rep
        trader_assets.append(ticker_data)
    
    # Combine all results
    trader_assets = pd.concat(trader_assets, ignore_index=True)
    
    # Calculate total profit/loss for each asset
    trader_assets['ProfitLoss'] = trader_assets['TradeAmount'] * trader_assets['AdjustedReturn'] / 100
    
    return trader_assets

def save_raw_data(df, trader_assets, output_dir):
    """Save raw data for verification"""
    # Save all columns from original data for selected traders
    raw_trades_path = os.path.join(output_dir, 'raw_trades.csv')
    df.to_csv(raw_trades_path, index=False)
    print(f"Saved raw trades to: {raw_trades_path}")
    
    # Save analyzed performance
    performance_path = os.path.join(output_dir, 'trader_performance.csv')
    trader_assets.to_csv(performance_path, index=False)
    print(f"Saved trader performance to: {performance_path}")
    
    # Create detailed summaries by quarter
    for quarter, traders in SELECTED_TRADERS.items():
        quarter_data = trader_assets[
            (trader_assets['Quarter'] == quarter) & 
            (trader_assets['Representative'].isin(traders))
        ]
        
        quarter_path = os.path.join(output_dir, f'{quarter}_summary.csv')
        quarter_data.to_csv(quarter_path, index=False)
        print(f"Saved {quarter} summary to: {quarter_path}")

def plot_individual_trader_analysis(trader_assets, df, trader, quarter, output_dir):
    """Create detailed analysis plots for each trader"""
    # Filter for specific trader and quarter
    trader_data = trader_assets[
        (trader_assets['Representative'] == trader) & 
        (trader_assets['Quarter'] == quarter)
    ]
    
    if len(trader_data) == 0:
        print(f"No data found for {trader} in {quarter}")
        return
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.patch.set_facecolor('#1C1C1C')  # Darker background like the image
    
    # Set title with more space and clear visibility
    fig.suptitle(f'{trader} Trading Analysis - {quarter}', 
                fontsize=20, y=1.02, color='white', fontweight='bold')
    
    # Special handling for Markwayne Mullin and Rob Bresnahan
    if trader in ['Markwayne Mullin', 'Rob Bresnahan']:
        # For bar chart: top 5 by absolute return, sorted by return value (highest first)
        trader_data_sorted = trader_data.sort_values('AdjustedReturn', ascending=False)
        plot_data_bars = trader_data_sorted.head(5)
        
        # For pie chart: top 5 by investment size
        plot_data_pie = trader_data.sort_values('TradeAmount', ascending=False)
        top_5_pie = plot_data_pie.head(5)
        others_sum = plot_data_pie.iloc[5:]['TradeAmount'].sum() if len(plot_data_pie) > 5 else 0
    else:
        plot_data_bars = trader_data.sort_values('AdjustedReturn', ascending=False)
        plot_data_pie = trader_data
        others_sum = 0
    
    # 1. Horizontal Bar Chart of Returns
    ax1.set_facecolor('#1C1C1C')  # Match image background
    
    # Custom colors for specific tickers
    def get_bar_color(ticker, return_value):
        if ticker == 'SKY':
            return '#3C99DC'  # Democrat blue
        elif ticker == 'AMZN':
            return '#E91D0E'  # Republican red
        else:
            return COLORS['democrat'] if return_value >= 0 else COLORS['republican']
    
    # Use custom colors for bars
    colors = [get_bar_color(ticker, ret) 
             for ticker, ret in zip(plot_data_bars['Ticker'], plot_data_bars['AdjustedReturn'])]
    
    bars = ax1.barh(plot_data_bars['Ticker'], plot_data_bars['AdjustedReturn'], 
                    color=colors, height=0.6)
    
    # Add value labels with larger font and clear visibility - all on right side
    for i, (v, trades) in enumerate(zip(plot_data_bars['AdjustedReturn'], 
                                      plot_data_bars['Transaction'])):
        # Format percentage with one decimal place
        label_text = f'{v:.1f}% ({len(trades)} trades)'
        
        # Position all text on the right side of the chart
        x_max = ax1.get_xlim()[1]
        
        ax1.text(x_max + 0.5, i, 
                label_text,
                va='center',
                ha='left',  # Always left-aligned from the right edge
                color='white',
                fontsize=12,
                fontweight='bold')
    
    # Enhance grid with more visible white lines
    ax1.grid(True, color='#FFFFFF', linestyle='-', alpha=0.2)  # Whiter, slightly transparent grid
    ax1.axvline(x=0, color='#FFD700', linestyle='--', alpha=0.5)  # Yellow SPY line
    
    # Set titles - show "Top 5" for both Rob and Mullin
    if trader in ['Rob Bresnahan', 'Markwayne Mullin']:
        ax1.set_title('Top 5 Returns by Stock', fontsize=16, pad=20, color='white', fontweight='bold')
        ax2.set_title('Top 5 Portfolio Allocation', fontsize=16, pad=20, color='white', fontweight='bold')
    else:
        ax1.set_title('Returns by Stock', fontsize=16, pad=20, color='white', fontweight='bold')
        ax2.set_title('Portfolio Allocation', fontsize=16, pad=20, color='white', fontweight='bold')
    
    # Make all tick labels white
    ax1.tick_params(axis='both', which='major', labelsize=12, colors='white')
    
    # Add extra padding on right side for labels
    x_min, x_max = ax1.get_xlim()
    ax1.set_xlim(x_min, x_max * 1.4)  # Add 40% padding on right side only
    
    # Make spines (border lines) more visible and white
    for spine in ax1.spines.values():
        spine.set_color('#FFFFFF')
        spine.set_alpha(0.2)
    
    # Ensure y-axis labels (ticker names) are white
    ax1.set_yticklabels(plot_data_bars['Ticker'], color='white', fontsize=12)
    
    # 2. Pie Chart of Portfolio Allocation
    ax2.set_facecolor('#1C1C1C')
    
    if trader in ['Markwayne Mullin', 'Rob Bresnahan']:
        sizes = list(top_5_pie['TradeAmount'])
        labels = [f"{ticker}\n${amount:,.0f}" 
                 for ticker, amount in zip(top_5_pie['Ticker'], top_5_pie['TradeAmount'])]
        if others_sum > 0:
            sizes.append(others_sum)
            labels.append(f"Others\n${others_sum:,.0f}")
    else:
        sizes = plot_data_pie['TradeAmount']
        labels = [f"{ticker}\n${amount:,.0f}" 
                 for ticker, amount in zip(plot_data_pie['Ticker'], plot_data_pie['TradeAmount'])]
    
    # Use brighter colors for better visibility
    pie_colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', '#8E8E8E']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=90, colors=pie_colors[:len(sizes)])
    
    # Enhance pie chart text
    plt.setp(autotexts, size=12, weight="bold", color='white')
    plt.setp(texts, size=12, color='white')
    
    # Add total portfolio value with enhanced visibility
    total_value = trader_data['TradeAmount'].sum()
    ax2.text(1.5, 1.1, f'Total Portfolio Value: ${total_value:,.0f}', 
             ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot with dark background
    plot_path = os.path.join(output_dir, 
                            f'trader_analysis_{trader.replace(" ", "_")}_{quarter}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                facecolor='#1C1C1C', edgecolor='none')
    print(f"Saved analysis plot for {trader} to: {plot_path}")
    plt.close()

def main():
    """Main function to run the analysis"""
    print("Starting Selected Traders Analysis...")
    
    # Setup output directory
    output_dir = setup_directories()
    
    try:
        # Load and filter data
        df = load_and_filter_data()
        
        # Analyze performance
        trader_assets = analyze_trader_performance(df)
        
        # Save raw data
        save_raw_data(df, trader_assets, output_dir)
        
        # Create individual plots for each trader
        for quarter, traders in SELECTED_TRADERS.items():
            for trader in traders:
                plot_individual_trader_analysis(trader_assets, df, trader, quarter, output_dir)
        
        print("Analysis complete. Check the output directory for results.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 