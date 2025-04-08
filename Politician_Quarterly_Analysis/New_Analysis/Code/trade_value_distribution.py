"""
Trade Value Distribution Analysis - 2025 Q1
------------------------------------------

This script analyzes the distribution of transaction values in politician trading data
to determine whether most trades are in smaller ranges (e.g., $1,001-$15,000) or if 
there are notable large transactions.

Visualizations include:
1. Histogram of transaction amounts
2. Box plot of transaction amounts by party
3. Breakdown of trades by value categories
4. Distribution comparison between Democrats and Republicans
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

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
    'positive': '#00C853',      # Green for buy/positive
    'negative': '#FF5252',      # Red for sell/negative
    'neutral': '#FFD700',       # Gold for reference lines
}

# Define transaction value ranges - REMOVED 0-1000 category
TRANSACTION_RANGES = [
    (1001, 15000),
    (15001, 50000),
    (50001, 100000),
    (100001, 250000),
    (250001, 500000),
    (500001, 1000000),
    (1000001, float('inf'))
]

# Create labels for the ranges - REMOVED 0-1000 category
RANGE_LABELS = [
    '$1,001-$15,000',
    '$15,001-$50,000',
    '$50,001-$100,000',
    '$100,001-$250,000',
    '$250,001-$500,000',
    '$500,001-$1,000,000',
    '$1,000,001+'
]


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
    
    # Quick verification of needed columns
    required_cols = ['TradeAmount', 'PartyName', 'TransactionType']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Filter out trades below $1,001 - NEW FILTER
    df = df[df['TradeAmount'] >= 1001]
    
    print(f"Loaded {len(df)} trades with transaction values above $1,000")
    return df


def analyze_trade_value_distribution(df):
    """Analyze trade value distribution and create summary statistics"""
    print("Analyzing trade value distribution...")
    
    # Calculate overall statistics
    overall_stats = {
        'mean': df['TradeAmount'].mean(),
        'median': df['TradeAmount'].median(),
        'min': df['TradeAmount'].min(),
        'max': df['TradeAmount'].max(),
        'std': df['TradeAmount'].std(),
        'total_value': df['TradeAmount'].sum(),
        'count': len(df)
    }
    
    # Calculate statistics by party
    party_stats = df.groupby('PartyName')['TradeAmount'].agg([
        'mean', 'median', 'min', 'max', 'std', 'sum', 'count'
    ]).reset_index()
    party_stats = party_stats.rename(columns={'sum': 'total_value'})
    
    # Calculate statistics by transaction type
    transaction_stats = df.groupby('TransactionType')['TradeAmount'].agg([
        'mean', 'median', 'min', 'max', 'std', 'sum', 'count'
    ]).reset_index()
    transaction_stats = transaction_stats.rename(columns={'sum': 'total_value'})
    
    # Categorize trades into value ranges
    def categorize_value(amount):
        for i, (lower, upper) in enumerate(TRANSACTION_RANGES):
            if lower <= amount <= upper:
                return RANGE_LABELS[i]
        return RANGE_LABELS[-1]  # If beyond all ranges (shouldn't happen)
    
    df['ValueCategory'] = df['TradeAmount'].apply(categorize_value)
    
    # Calculate distribution by value category
    category_counts = df.groupby(['ValueCategory', 'PartyName']).size().unstack(fill_value=0)
    
    # Make sure all categories are present, even if count is zero
    for label in RANGE_LABELS:
        if label not in category_counts.index:
            category_counts.loc[label] = 0
    
    # Sort by the defined order of ranges
    category_counts = category_counts.reindex(RANGE_LABELS)
    
    # Calculate percentage of trades in each category
    category_percentages = category_counts.copy()
    for col in category_percentages.columns:
        category_percentages[col] = (category_percentages[col] / category_percentages[col].sum()) * 100
    
    # Calculate distribution by value category and transaction type
    type_category_counts = df.groupby(['ValueCategory', 'TransactionType']).size().unstack(fill_value=0)
    type_category_counts = type_category_counts.reindex(RANGE_LABELS)
    
    print("\nOverall trade value statistics:")
    print(f"Mean trade value: ${overall_stats['mean']:,.2f}")
    print(f"Median trade value: ${overall_stats['median']:,.2f}")
    print(f"Standard deviation: ${overall_stats['std']:,.2f}")
    print(f"Range: ${overall_stats['min']:,.2f} to ${overall_stats['max']:,.2f}")
    print(f"Total trade value: ${overall_stats['total_value']:,.2f}")
    print(f"Total number of trades: {overall_stats['count']}")
    
    return overall_stats, party_stats, transaction_stats, category_counts, category_percentages, type_category_counts


def plot_trade_value_histogram(df, overall_stats):
    """Create a histogram of trade values"""
    print("Creating trade value histogram...")
    
    # Create figure with better dimensions
    plt.figure(figsize=(16, 9))
    
    # Prepare data by party
    dem_values = df[df['PartyName'] == 'Democrats']['TradeAmount']
    rep_values = df[df['PartyName'] == 'Republicans']['TradeAmount']
    
    # Define logarithmic bins for better visualization
    min_value = max(1, df['TradeAmount'].min())  # Avoid log(0)
    max_value = df['TradeAmount'].max()
    log_bins = np.logspace(np.log10(min_value), np.log10(max_value), 40)
    
    # Create histogram with party colors
    plt.hist([dem_values, rep_values], bins=log_bins, stacked=True, 
             color=[COLORS['democrat'], COLORS['republican']], 
             alpha=0.8, edgecolor='white', label=['Democrats', 'Republicans'])
    
    plt.xscale('log')
    
    # Add vertical lines for key statistics with improved styling
    plt.axvline(x=overall_stats['median'], color=COLORS['neutral'], linestyle='--', linewidth=2, 
                label=f"Median: ${overall_stats['median']:,.0f}")
    plt.axvline(x=overall_stats['mean'], color='white', linestyle='--', linewidth=2, 
                label=f"Mean: ${overall_stats['mean']:,.0f}")
    
    # Add labels and formatting
    plt.title('Distribution of Trade Values - 2025 Q1', fontsize=20, pad=20)
    plt.xlabel('Trade Value ($ - Log Scale)', fontsize=14, labelpad=10)
    plt.ylabel('Number of Trades', fontsize=14, labelpad=10)
    plt.grid(axis='both', alpha=0.3)
    
    # Format x-axis with dollar amounts
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # Add annotations for common transaction ranges with improved styling
    common_ranges = [1000, 15000, 50000, 100000, 250000, 500000, 1000000]
    for value in common_ranges:
        if min_value <= value <= max_value:
            plt.axvline(x=value, color='gray', linestyle=':', alpha=0.4)
            plt.text(value, plt.gca().get_ylim()[1] * 0.92, f"${value:,}", 
                    rotation=90, ha='right', va='top', color='gray', alpha=0.8,
                    fontsize=10, bbox=dict(facecolor=COLORS['background'], alpha=0.6, pad=1))
    
    # Add enhanced legend
    legend = plt.legend(fontsize=12, loc='upper right')
    legend.get_frame().set_alpha(0.9)
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'trade_value_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved trade value histogram to: {output_path}")
    plt.close()


def plot_value_category_distribution(category_counts, category_percentages):
    """Create a horizontal bar chart showing the distribution of trades by value category"""
    print("Creating trade value category distribution chart...")
    
    # Create figure with two subplots - use horizontal layout to match other charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle('Trade Value Category Distribution - 2025 Q1', fontsize=22, y=0.98)
    
    # Reorder categories to have largest values at the top
    # This makes it consistent with the other horizontal bar charts
    reordered_labels = RANGE_LABELS.copy()
    category_counts = category_counts.reindex(reordered_labels)
    category_percentages = category_percentages.reindex(reordered_labels)
    
    # Plot 1: Count of trades by value category and party - HORIZONTAL
    category_counts.plot(kind='barh', ax=ax1, color=[COLORS['democrat'], COLORS['republican']])
    
    # Add labels and formatting
    ax1.set_title('Number of Trades by Value Category', fontsize=16)
    ax1.set_xlabel('Number of Trades', fontsize=14)
    ax1.set_ylabel('Trade Value Category', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add count labels next to each bar
    for i, container in enumerate(ax1.containers):
        party_color = COLORS['democrat'] if i == 0 else COLORS['republican']
        for j, val in enumerate(container):
            width = val.get_width()
            label_text = f"{int(width)}"
            ax1.text(width + 2, val.get_y() + val.get_height()/2, 
                    label_text, ha='left', va='center', 
                    color=party_color, fontweight='bold', fontsize=10)
    
    # Plot 2: Percentage of trades by value category and party - HORIZONTAL
    category_percentages.plot(kind='barh', ax=ax2, color=[COLORS['democrat'], COLORS['republican']])
    
    # Add labels and formatting
    ax2.set_title('Percentage of Trades by Value Category', fontsize=16)
    ax2.set_xlabel('Percentage of Trades (%)', fontsize=14)
    ax2.set_ylabel('Trade Value Category', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels next to each bar
    for i, container in enumerate(ax2.containers):
        party_color = COLORS['democrat'] if i == 0 else COLORS['republican']
        for j, val in enumerate(container):
            width = val.get_width()
            label_text = f"{width:.1f}%"
            ax2.text(width + 0.5, val.get_y() + val.get_height()/2, 
                    label_text, ha='left', va='center', 
                    color=party_color, fontweight='bold', fontsize=10)
    
    # Add legend with better position
    ax1.legend(title='Party', loc='upper right')
    ax2.legend(title='Party', loc='upper right')
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'trade_value_category_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved trade value category distribution chart to: {output_path}")
    plt.close()


def plot_transaction_type_by_value(df):
    """Create a stacked bar chart showing buys vs. sells by value category"""
    print("Creating transaction type by value category chart...")
    
    # Calculate the counts by transaction type and value category
    type_value_counts = pd.crosstab(
        index=df['ValueCategory'], 
        columns=df['TransactionType'], 
        values=df['TradeAmount'],
        aggfunc='count'
    ).fillna(0)
    
    # Make sure all categories are present, even if count is zero
    for label in RANGE_LABELS:
        if label not in type_value_counts.index:
            type_value_counts.loc[label] = 0
    
    # Sort by the defined order of ranges
    type_value_counts = type_value_counts.reindex(RANGE_LABELS)
    
    # Calculate percentages
    type_value_percentages = type_value_counts.copy()
    type_value_percentages['total'] = type_value_percentages.sum(axis=1)
    for col in ['purchase', 'sale']:
        if col in type_value_percentages.columns:
            type_value_percentages[col] = (type_value_percentages[col] / type_value_percentages['total']) * 100
    
    type_value_percentages = type_value_percentages.drop('total', axis=1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle('Transaction Types by Value Category - 2025 Q1', fontsize=22, y=0.98)
    
    # Use less bright colors for buy/sell
    buy_color = '#00CC00'  # Less bright green
    sell_color = '#E60000'  # Less bright red
    
    # Plot 1: Count of trades by value category and transaction type
    type_value_counts.plot(kind='barh', ax=ax1, stacked=True, color=[buy_color, sell_color])
    
    # Add labels and formatting
    ax1.set_title('Number of Trades by Value and Type', fontsize=16)
    ax1.set_xlabel('Number of Trades', fontsize=14)
    ax1.set_ylabel('Trade Value Category', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # Get maximum total width for padding on right side
    total_max = type_value_counts.sum(axis=1).max()
    
    # Add labels to the right side of the bars in "buy : sell" format
    for i, (idx, row) in enumerate(type_value_counts.iterrows()):
        buy_count = row.get('purchase', 0)
        sell_count = row.get('sale', 0)
        
        # Format as "buy : sell"
        label_text = f"{int(buy_count)} : {int(sell_count)}"
        
        # Position label to the right of the total bar
        total_width = buy_count + sell_count
        ax1.text(
            total_width + (total_max * 0.03),  # Add padding to the right
            i,  # Y position (bar index)
            label_text,
            ha='left', va='center',
            color=COLORS['text'], fontsize=12, fontweight='bold'
        )
    
    # Expand x-axis to make room for labels
    x_max = total_max * 1.2
    ax1.set_xlim(0, x_max)
    
    # Plot 2: Percentage of trades by value category and transaction type
    type_value_percentages.plot(kind='barh', ax=ax2, stacked=True, color=[buy_color, sell_color])
    
    # Add labels and formatting
    ax2.set_title('Percentage of Buy vs. Sell by Value', fontsize=16)
    ax2.set_xlabel('Percentage of Trades (%)', fontsize=14)
    ax2.set_ylabel('Trade Value Category', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels in center of each segment
    buy_pct_bars = ax2.containers[0]
    sell_pct_bars = ax2.containers[1]
    
    # Add labels for buy percentage bars
    for i, bar in enumerate(buy_pct_bars):
        width = bar.get_width()
        if width > 5:  # Only show if wide enough
            # Position label in the center of the bar segment
            ax2.text(
                width / 2,  # Position in middle of bar
                bar.get_y() + bar.get_height() / 2,  # Vertical center
                f"{width:.1f}%",  # The percentage
                ha='center', va='center',  # Alignment
                color='white', fontweight='bold'  # Styling
            )
    
    # Add labels for sell percentage bars
    for i, bar in enumerate(sell_pct_bars):
        width = bar.get_width()
        if width > 5:  # Only show if wide enough
            # Position label in the center of the sell segment
            ax2.text(
                buy_pct_bars[i].get_width() + width / 2,  # Position in middle of segment
                bar.get_y() + bar.get_height() / 2,  # Vertical center
                f"{width:.1f}%",  # The percentage
                ha='center', va='center',  # Alignment
                color='white', fontweight='bold'  # Styling
            )
    
    # Create custom legend patches
    buy_patch = mpatches.Patch(color=buy_color, label='Buy')
    sell_patch = mpatches.Patch(color=sell_color, label='Sell')
    
    # Update the legend
    ax1.legend(handles=[buy_patch, sell_patch], loc='upper right')
    ax2.legend(handles=[buy_patch, sell_patch], loc='upper right')
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'transaction_type_by_value.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved transaction type by value chart to: {output_path}")
    plt.close()


def plot_trade_value_boxplot(df):
    """Create a box plot of trade values by party"""
    print("Creating trade value box plot by party...")
    
    # Create figure
    plt.figure(figsize=(16, 9))
    
    # Set up the data
    dem_values = df[df['PartyName'] == 'Democrats']['TradeAmount']
    rep_values = df[df['PartyName'] == 'Republicans']['TradeAmount']
    
    # Calculate key statistics for each party
    dem_stats = {
        'median': dem_values.median(),
        'mean': dem_values.mean(),
        'q25': dem_values.quantile(0.25),
        'q75': dem_values.quantile(0.75)
    }
    
    rep_stats = {
        'median': rep_values.median(),
        'mean': rep_values.mean(),
        'q25': rep_values.quantile(0.25),
        'q75': rep_values.quantile(0.75)
    }
    
    # Create a custom box plot with better styling
    boxprops = dict(linestyle='-', linewidth=2, alpha=0.8)
    medianprops = dict(linestyle='-', linewidth=2.5, color='white')
    whiskerprops = dict(linestyle='-', linewidth=1.5, alpha=0.7)
    capprops = dict(linestyle='-', linewidth=1.5)
    flierprops = dict(marker='o', markerfacecolor='white', alpha=0.5, 
                    markersize=5, markeredgecolor='none')
    
    # Create the box plot with enhanced styling
    sns.boxplot(data=[dem_values, rep_values], 
               boxprops=boxprops, medianprops=medianprops, 
               whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops,
               palette=[COLORS['democrat'], COLORS['republican']], width=0.5)
    
    # Use logarithmic scale for y-axis to see the distribution better
    plt.yscale('log')
    
    # Add horizontal lines for common transaction ranges with improved styling
    common_ranges = [1000, 15000, 50000, 100000, 250000, 500000, 1000000]
    for value in common_ranges:
        min_value = df['TradeAmount'].min()
        max_value = df['TradeAmount'].max()
        if min_value <= value <= max_value:
            plt.axhline(y=value, color='gray', linestyle=':', alpha=0.5)
            plt.text(1.8, value, f"${value:,}", 
                    ha='left', va='center', color='gray', alpha=0.8,
                    fontsize=10, bbox=dict(facecolor=COLORS['background'], alpha=0.6, pad=1))
    
    # Add statistics annotations with improved styling
    # Democrats
    plt.text(0, dem_stats['median'] * 1.5, f"Median: ${dem_stats['median']:,.0f}", 
            ha='center', color='white', fontweight='bold', fontsize=11,
            bbox=dict(facecolor=COLORS['democrat'], alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.text(0, dem_stats['mean'] * 2.5, f"Mean: ${dem_stats['mean']:,.0f}", 
            ha='center', color='white', fontweight='bold', fontsize=11,
            bbox=dict(facecolor=COLORS['democrat'], alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Republicans
    plt.text(1, rep_stats['median'] * 1.5, f"Median: ${rep_stats['median']:,.0f}", 
            ha='center', color='white', fontweight='bold', fontsize=11,
            bbox=dict(facecolor=COLORS['republican'], alpha=0.7, boxstyle='round,pad=0.5'))
    
    plt.text(1, rep_stats['mean'] * 2.5, f"Mean: ${rep_stats['mean']:,.0f}", 
            ha='center', color='white', fontweight='bold', fontsize=11,
            bbox=dict(facecolor=COLORS['republican'], alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add labels and formatting
    plt.title('Trade Value Distribution by Party - 2025 Q1', fontsize=20, pad=20)
    plt.ylabel('Trade Value ($ - Log Scale)', fontsize=14, labelpad=10)
    plt.xticks([0, 1], ['Democrats', 'Republicans'], fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Format y-axis with dollar amounts
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'trade_value_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved trade value box plot to: {output_path}")
    plt.close()


def main():
    """Main function to run the trade value distribution analysis"""
    print("Starting Trade Value Distribution Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Analyze trade value distribution
        overall_stats, party_stats, transaction_stats, category_counts, category_percentages, type_category_counts = analyze_trade_value_distribution(df)
        
        # Create visualizations
        plot_trade_value_histogram(df, overall_stats)
        plot_trade_value_boxplot(df)
        plot_value_category_distribution(category_counts, category_percentages)
        plot_transaction_type_by_value(df)
        
        print("Trade value distribution analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during trade value distribution analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 