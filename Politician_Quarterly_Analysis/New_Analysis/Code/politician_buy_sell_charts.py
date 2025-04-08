"""
Individual Politician Buy/Sell Volume Analysis - 2025 Q1
--------------------------------------------------------

This script creates two horizontal bar charts showing:
1. Buy volume by individual politician
2. Sell volume by individual politician

Charts use the same style as the portfolio performance chart with:
- Republicans in red, Democrats in blue
- Horizontal bar format with largest volume at top
- Dark background theme with clear labels
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import math

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')  # Use the enriched data
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# Define custom color scheme - dark theme with blue/red political colors
COLORS = {
    'background': '#121212',    # Dark background
    'text': '#FFFFFF',          # White text
    'grid': '#333333',          # Dark grid
    'democrat': '#3C99DC',      # Democrat blue
    'republican': '#E91D0E',    # Republican red
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
        'font.weight': 'normal',
        'legend.facecolor': COLORS['background'],
        'legend.edgecolor': COLORS['grid'],
        'figure.titlesize': 18,
        'axes.titlesize': 16,
        'axes.labelsize': 14
    })


def load_enriched_data():
    """Load the enriched trading data"""
    input_file = os.path.join(INPUT_DIR, 'enriched_2025q1_trading_data.csv')
    print(f"Loading enriched data from {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Enriched data file not found at {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    
    # Quick verification of needed columns
    required_cols = ['TransactionType', 'TradeAmount', 'PartyName', 'Representative']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    print(f"Loaded {len(df)} trades")
    return df


def prepare_politician_volume_data(df):
    """Prepare data for politician buy/sell volume charts"""
    print("Preparing politician buy/sell data...")
    
    # Group by politician, party and transaction type
    politician_volume = df.groupby(['Representative', 'PartyName', 'TransactionType'])['TradeAmount'].sum().reset_index()
    
    # Create separate buy and sell dataframes
    buy_volume = politician_volume[politician_volume['TransactionType'] == 'purchase'].copy()
    sell_volume = politician_volume[politician_volume['TransactionType'] == 'sale'].copy()
    
    # Sort by volume (descending)
    buy_volume = buy_volume.sort_values('TradeAmount', ascending=False)
    sell_volume = sell_volume.sort_values('TradeAmount', ascending=False)
    
    # Calculate total trade count for each politician's buys and sells
    buy_counts = df[df['TransactionType'] == 'purchase'].groupby('Representative').size().reset_index(name='TradeCount')
    sell_counts = df[df['TransactionType'] == 'sale'].groupby('Representative').size().reset_index(name='TradeCount')
    
    # Add trade counts to the volume data
    buy_volume = buy_volume.merge(buy_counts, on='Representative')
    sell_volume = sell_volume.merge(sell_counts, on='Representative')
    
    return buy_volume, sell_volume


def plot_volume_chart(volume_data, transaction_type, output_dir, max_politicians=45):
    """
    Create a horizontal bar chart showing politician trading volumes
    Similar to the portfolio performance chart in style and layout
    """
    chart_title = 'Purchase' if transaction_type == 'buy' else 'Sale'
    print(f"Creating {chart_title} volume chart...")
    
    # Take top N politicians by volume to keep chart readable
    if len(volume_data) > max_politicians:
        plot_data = volume_data.head(max_politicians).copy()
        print(f"Showing top {max_politicians} politicians by {transaction_type} volume")
    else:
        plot_data = volume_data.copy()
        print(f"Showing all {len(volume_data)} politicians with {transaction_type}s")
    
    # Calculate figure height based on number of politicians
    fig_height = max(10, len(plot_data) * 0.45)  # Same height calculation as performance chart
    plt.figure(figsize=(16, fig_height))
    
    # Reverse order for display (matplotlib plots from bottom to top)
    plot_data = plot_data.iloc[::-1].reset_index(drop=True)
    
    # Create positions for bars
    positions = np.arange(len(plot_data))
    
    # Create color map based on party
    colors = [COLORS['democrat'] if party == 'Democrats' else COLORS['republican'] 
              for party in plot_data['PartyName']]
    
    # Plot horizontal bars - display in millions for the chart
    bars = plt.barh(positions, plot_data['TradeAmount'] / 1e6, height=0.7, color=colors)
    
    # Add labels and title
    plt.title(f'Politician {chart_title} Volume - 2025-Q1', fontsize=22, pad=20)
    plt.xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
    plt.ylabel('Politician', fontsize=14)
    
    # Add subtitle
    plt.figtext(0.5, 0.02, 
               f'Each bar shows the total {chart_title.lower()} volume in politician stock portfolio', 
               fontsize=12, ha='center', color=COLORS['text'])
    
    # Create party legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['democrat'], label='Democrat'),
        Patch(facecolor=COLORS['republican'], label='Republican')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Format y-axis with politician names
    plt.yticks(positions, plot_data['Representative'])
    
    # Calculate x-axis padding
    max_volume = plot_data['TradeAmount'].max() / 1e6
    x_max = max_volume * 1.2  # Add 20% padding
    
    # Update x-axis limits
    plt.xlim(left=0, right=x_max)
    
    # Add gridlines at regular intervals
    grid_step = 1  # Grid lines every $1M
    if max_volume > 20:
        grid_step = 5  # Every $5M for larger values
    elif max_volume > 10:
        grid_step = 2  # Every $2M for medium values
    
    grid_ticks = np.arange(0, math.ceil(x_max / grid_step) * grid_step + grid_step, grid_step)
    plt.xticks(grid_ticks)
    
    # Make the grid more visible
    plt.grid(axis='x', linestyle='--', alpha=0.7, which='major')
    
    # Add value labels and trade counts
    for i, (value, trades) in enumerate(zip(plot_data['TradeAmount'], plot_data['TradeCount'])):
        # Format value based on size
        if value < 1e6:  # Less than 1 million
            value_in_thousands = value / 1e3
            formatted_value = f"${value_in_thousands:.1f}K"
        else:  # 1 million or more
            value_in_millions = value / 1e6
            formatted_value = f"${value_in_millions:.1f}M"
        
        # Position labels at end of bars - use the value in millions for positioning
        label_x = value / 1e6 + 0.2
        plt.text(label_x, i, f"{formatted_value} ({trades} trades)", 
                va='center', color=COLORS['text'], fontsize=10)
    
    # Add extra space below the bottom-most bar
    plt.ylim(-1, len(plot_data))
    
    # Add a source watermark
    plt.figtext(0.5, 0.005, 'Data source: Quiver Quantitative', 
                fontsize=8, ha='center', color=COLORS['text'])
    
    # Save the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_path = os.path.join(output_dir, f'politician_{transaction_type}_volume_2025q1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved {transaction_type} volume chart to: {output_path}")
    plt.close()
    
    return output_path


def main():
    """Main function to create buy/sell volume charts by politician"""
    print("Starting Individual Politician Buy/Sell Volume Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Prepare buy/sell volume data
        buy_volume, sell_volume = prepare_politician_volume_data(df)
        
        # Create charts
        plot_volume_chart(buy_volume, 'buy', OUTPUT_DIR)
        plot_volume_chart(sell_volume, 'sell', OUTPUT_DIR)
        
        print("Buy/Sell volume charts complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during chart creation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 