"""
Transaction Timeline Analysis - 2025 Q1
--------------------------------------

This script creates a timeline visualization of buy and sell transactions,
showing trading volume patterns over time throughout Q1 2025.

The visualization includes:
1. Daily transaction counts separated by buy/sell
2. Rolling average trend lines
3. Key date annotations for significant trading activity
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime, timedelta

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')  # Use the enriched data
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# Define custom color scheme - dark theme with green/red buy/sell colors
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'buy': '#00CC00',           # Green for buy
    'sell': '#E60000',          # Red for sell
    'buy_area': '#00CC0033',    # Semi-transparent green for area
    'sell_area': '#E6000033',   # Semi-transparent red for area
    'rolling_buy': '#00FF44',   # Bright green for rolling average line
    'rolling_sell': '#FF5555',  # Bright red for rolling average line
    'annotation': '#FFFF99',    # Yellow for annotations
}

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
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'legend.facecolor': COLORS['background'],
        'legend.edgecolor': COLORS['grid'],
        'figure.titlesize': 20,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

def load_enriched_data():
    """Load the enriched trading data"""
    input_file = os.path.join(INPUT_DIR, 'enriched_2025q1_trading_data.csv')
    print(f"Loading enriched data from {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Enriched data file not found at {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    
    # Explicitly check that TransactionDate column exists
    required_cols = ['TransactionDate', 'TransactionType']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Convert TransactionDate to datetime
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    
    # Filter for Q1 2025 only using TransactionDate
    q1_start = pd.Timestamp('2025-01-01')
    q1_end = pd.Timestamp('2025-03-31')
    df = df[(df['TransactionDate'] >= q1_start) & (df['TransactionDate'] <= q1_end)]
    
    print(f"Loaded {len(df)} transactions for Q1 2025")
    return df

def analyze_transaction_timeline(df):
    """Analyze transactions over time to create timeline data"""
    print("Analyzing transaction timeline...")
    
    # Group by date and transaction type
    daily_counts = df.groupby([pd.Grouper(key='TransactionDate', freq='D'), 'TransactionType']).size().unstack(fill_value=0)
    
    # Make sure both purchase and sale columns exist
    if 'purchase' not in daily_counts.columns:
        daily_counts['purchase'] = 0
    if 'sale' not in daily_counts.columns:
        daily_counts['sale'] = 0
    
    # Reset index to make date a column
    daily_counts = daily_counts.reset_index()
    
    # Find days with notable activity (peaks or unusual patterns)
    total_daily = daily_counts['purchase'] + daily_counts['sale']
    mean_daily = total_daily.mean()
    std_daily = total_daily.std()
    
    # Identify peaks (days with activity > mean + 1.5*std)
    peak_threshold = mean_daily + 1.5 * std_daily
    peak_days = daily_counts[total_daily > peak_threshold].copy()
    
    # Sort peak days by total volume for annotation purposes
    peak_days['total'] = peak_days['purchase'] + peak_days['sale']
    peak_days = peak_days.sort_values('total', ascending=False)
    
    # Find top traded asset for each peak day
    peak_days['top_asset'] = ""
    peak_days['top_asset_count'] = 0
    peak_days['top_asset_sector'] = ""
    
    for i, row in peak_days.iterrows():
        date = row['TransactionDate']
        # Filter transactions for this specific date
        day_transactions = df[df['TransactionDate'] == date]
        
        # Group by ticker and count
        ticker_counts = day_transactions.groupby('Ticker').size().reset_index(name='count')
        if not ticker_counts.empty:
            # Get the top ticker
            top_ticker = ticker_counts.sort_values('count', ascending=False).iloc[0]
            
            # Find sector for this ticker
            ticker_sector = day_transactions[day_transactions['Ticker'] == top_ticker['Ticker']]['Sector'].iloc[0] \
                            if 'Sector' in day_transactions.columns else "Unknown"
            
            # Update peak_days with top asset info
            peak_days.at[i, 'top_asset'] = top_ticker['Ticker']
            peak_days.at[i, 'top_asset_count'] = top_ticker['count']
            peak_days.at[i, 'top_asset_sector'] = ticker_sector
    
    print(f"Average daily transactions: {mean_daily:.1f}")
    print(f"Found {len(peak_days)} days with notable trading activity")
    
    return daily_counts, peak_days

def plot_transaction_timeline(daily_counts, peak_days):
    """Create a timeline visualization of buy and sell transactions"""
    print("Creating transaction timeline visualization...")
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Plot daily counts with semi-transparent areas
    plt.fill_between(daily_counts['TransactionDate'], daily_counts['purchase'], 
                    alpha=0.3, color=COLORS['buy_area'], label='_nolegend_')
    plt.fill_between(daily_counts['TransactionDate'], daily_counts['sale'], 
                    alpha=0.3, color=COLORS['sell_area'], label='_nolegend_')
    
    # Plot daily counts with markers
    plt.plot(daily_counts['TransactionDate'], daily_counts['purchase'], 
            marker='o', markersize=4, linestyle='-', linewidth=1.5, 
            color=COLORS['buy'], label='Daily Buys')
    plt.plot(daily_counts['TransactionDate'], daily_counts['sale'], 
            marker='o', markersize=4, linestyle='-', linewidth=1.5, 
            color=COLORS['sell'], label='Daily Sells')
    
    # Add annotations for peak days (limit to top 5 to avoid clutter)
    for i, row in peak_days.head(5).iterrows():
        date = row['TransactionDate']
        total = row['total']
        buy_count = row['purchase']
        sell_count = row['sale']
        top_asset = row['top_asset']
        top_asset_count = row['top_asset_count']
        top_asset_sector = row['top_asset_sector']
        
        # Position annotation with a slight offset
        x_pos = date
        y_pos = max(buy_count, sell_count) + 3
        
        # Add annotation text with top asset information
        annotation_text = f"{date.strftime('%b %d')}: {int(total)} trades\n"
        annotation_text += f"({int(buy_count)} buys, {int(sell_count)} sells)\n"
        if top_asset:
            annotation_text += f"Top: {top_asset} ({int(top_asset_count)} trades)"
            if top_asset_sector:
                annotation_text += f"\n{top_asset_sector}"
        
        plt.annotate(annotation_text,
                    xy=(date, max(buy_count, sell_count)),
                    xytext=(x_pos, y_pos),
                    arrowprops=dict(facecolor=COLORS['annotation'], shrink=0.05, width=1.5, headwidth=8),
                    ha='center', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc=COLORS['background'], ec=COLORS['annotation'], alpha=0.9),
                    color=COLORS['annotation'],
                    fontsize=10)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # Monday as major tick
    plt.gcf().autofmt_xdate()  # Rotate date labels
    
    # Format y-axis to use integers only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add labels and title
    plt.title('Daily Trading Activity - Q1 2025', fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=14, labelpad=10)
    plt.ylabel('Number of Transactions', fontsize=14, labelpad=10)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend with better positioning
    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                       fancybox=True, shadow=True, ncol=2, fontsize=12)
    
    # Enhance the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Add quarter labels to mark months
    for month in [1, 2, 3]:
        month_start = pd.Timestamp(f'2025-{month:02d}-01')
        month_label = month_start.strftime('%B')
        plt.axvline(x=month_start, color=COLORS['grid'], linestyle='--', alpha=0.5)
        plt.text(month_start + timedelta(days=15), plt.ylim()[1] * 0.95, 
                month_label, ha='center', va='top', 
                bbox=dict(facecolor=COLORS['background'], alpha=0.7, edgecolor='none'))
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'transaction_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved transaction timeline visualization to: {output_path}")
    plt.close()
    
    return output_path

def create_monthly_breakdown(df):
    """Create a monthly breakdown of buy vs. sell transactions"""
    print("Creating monthly transaction breakdown...")
    
    # Group by month and transaction type
    df['Month'] = df['TransactionDate'].dt.month
    monthly_counts = df.groupby(['Month', 'TransactionType']).size().unstack(fill_value=0)
    
    # Calculate total trades per month
    monthly_counts['total'] = monthly_counts.sum(axis=1)
    
    # Add month names
    month_names = {1: 'January', 2: 'February', 3: 'March'}
    monthly_counts['MonthName'] = monthly_counts.index.map(month_names)
    
    # Create figure for monthly breakdown
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    x = np.arange(len(monthly_counts))
    width = 0.35
    
    plt.bar(x - width/2, monthly_counts['purchase'], width, label='Buy', color=COLORS['buy'], alpha=0.9)
    plt.bar(x + width/2, monthly_counts['sale'], width, label='Sell', color=COLORS['sell'], alpha=0.9)
    
    # Add count labels on top of bars
    for i, count in enumerate(monthly_counts['purchase']):
        plt.text(i - width/2, count + 5, str(count), ha='center', va='bottom', color=COLORS['text'], fontweight='bold')
    
    for i, count in enumerate(monthly_counts['sale']):
        plt.text(i + width/2, count + 5, str(count), ha='center', va='bottom', color=COLORS['text'], fontweight='bold')
    
    # Add labels and title
    plt.title('Monthly Trading Volume - Q1 2025', fontsize=20, pad=20)
    plt.xlabel('Month', fontsize=14, labelpad=10)
    plt.ylabel('Number of Transactions', fontsize=14, labelpad=10)
    
    # Set x-axis ticks and labels
    plt.xticks(x, monthly_counts['MonthName'])
    
    # Add totals as text above each month
    for i, total in enumerate(monthly_counts['total']):
        plt.text(i, monthly_counts[['purchase', 'sale']].max(axis=1)[i+1] + 30, 
                f"Total: {total}", ha='center', va='bottom', 
                color=COLORS['text'], fontsize=12)
    
    # Add grid and legend
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              fancybox=True, shadow=True, ncol=2, fontsize=12)
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'monthly_transaction_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved monthly breakdown visualization to: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Main function to run the transaction timeline analysis"""
    print("Starting Transaction Timeline Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Analyze transaction timeline
        daily_counts, peak_days = analyze_transaction_timeline(df)
        
        # Create visualizations
        plot_transaction_timeline(daily_counts, peak_days)
        create_monthly_breakdown(df)
        
        print("Transaction timeline analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during transaction timeline analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 