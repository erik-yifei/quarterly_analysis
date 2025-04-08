"""
Top 5 Stocks Traded by Party - 2025 Q1
--------------------------------------

This script analyzes the most traded stocks by volume for Democrats and Republicans
and creates visualizations showing:
1. Top 5 stocks traded by Democrats 
2. Top 5 stocks traded by Republicans
3. Sector information for each stock

The charts follow the same style as other portfolio performance charts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
}

# Sector-based color mapping for consistent coloring
SECTOR_COLORS = {
    'Information Technology': '#58D68D',  # Green
    'Health Care': '#3498DB',            # Blue
    'Consumer Discretionary': '#E74C3C',  # Red
    'Financials': '#F4D03F',            # Yellow
    'Communication Services': '#9B59B6',  # Purple
    'Industrials': '#E67E22',           # Orange
    'Energy': '#2C3E50',                # Dark Blue
    'Consumer Staples': '#1ABC9C',       # Teal
    'Materials': '#D35400',             # Dark Orange
    'Real Estate': '#CD6155',           # Pink
    'Utilities': '#7F8C8D',             # Gray
    'Unknown': '#95A5A6'                # Light Gray
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
    required_cols = ['Ticker', 'TradeAmount', 'PartyName', 'Sector']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Handle missing sector data
    df['Sector'] = df['Sector'].fillna('Unknown')
    print(f"Loaded {len(df)} trades with ticker information")
    
    return df


def get_top_stocks_by_party(df, top_n=5):
    """Get the top N most traded stocks by volume for each party"""
    print(f"Finding top {top_n} stocks by trading volume for each party...")
    
    # Group by party and ticker
    party_ticker_volume = df.groupby(['PartyName', 'Ticker'])['TradeAmount'].sum().reset_index()
    
    # Get ticker sector mapping
    ticker_sector = df.groupby('Ticker')['Sector'].first().to_dict()
    
    # Add sector information
    party_ticker_volume['Sector'] = party_ticker_volume['Ticker'].map(ticker_sector)
    
    # Get transaction counts for each ticker by party
    ticker_counts = df.groupby(['PartyName', 'Ticker']).size().reset_index(name='TradeCount')
    party_ticker_volume = party_ticker_volume.merge(ticker_counts, on=['PartyName', 'Ticker'])
    
    # Get top N for each party
    democrats = party_ticker_volume[party_ticker_volume['PartyName'] == 'Democrats']
    republicans = party_ticker_volume[party_ticker_volume['PartyName'] == 'Republicans']
    
    dem_top = democrats.nlargest(top_n, 'TradeAmount')
    rep_top = republicans.nlargest(top_n, 'TradeAmount')
    
    print("\nTop stocks for Democrats:")
    for _, row in dem_top.iterrows():
        print(f"{row['Ticker']} ({row['Sector']}): ${row['TradeAmount']:,.0f}")
    
    print("\nTop stocks for Republicans:")
    for _, row in rep_top.iterrows():
        print(f"{row['Ticker']} ({row['Sector']}): ${row['TradeAmount']:,.0f}")
    
    return dem_top, rep_top


def plot_top_stocks_by_party(dem_stocks, rep_stocks):
    """Create horizontal bar chart of top stocks by party"""
    print("Creating top stocks by party visualization...")
    
    # Create figure with two subplots (one for each party)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))  # Increased figure size
    fig.suptitle('Top 5 Stocks Traded by Party - 2025 Q1', fontsize=22, y=0.98)
    
    # Function to create the bar chart for a party
    def create_party_chart(ax, data, party, party_color):
        # Sort by trade amount descending
        data = data.sort_values('TradeAmount', ascending=True)
        
        # Get colors based on sectors
        sector_color_map = {sector: SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) 
                           for sector in data['Sector'].unique()}
        
        # Create bars colored by sector
        bar_colors = [sector_color_map[sector] for sector in data['Sector']]
        
        # Create positions for bars
        positions = np.arange(len(data))
        
        # Plot horizontal bars with increased alpha for better appearance
        bars = ax.barh(positions, data['TradeAmount'] / 1e6, height=0.7, color=bar_colors, alpha=0.9)
        
        # Set y-ticks and labels - just ticker names, no sector labels
        ax.set_yticks(positions)
        ax.set_yticklabels(data['Ticker'], fontsize=13, fontweight='bold')
        
        # Add value labels to the right of each bar with improved formatting
        for i, (amount, count) in enumerate(zip(data['TradeAmount'], data['TradeCount'])):
            # Format by thousands or millions
            if amount < 1e6:
                formatted_value = f"${amount/1e3:.1f}K"
            else:
                formatted_value = f"${amount/1e6:.1f}M"
            
            # Add better positioned label with background highlight
            ax.text(amount/1e6 + 0.5, i, f"{formatted_value} ({count} trades)", 
                   ha='left', va='center', color=COLORS['text'], fontsize=11, 
                   bbox=dict(facecolor=COLORS['background'], alpha=0.7, edgecolor=None, pad=2))
        
        # Add title and labels
        ax.set_title(party, color=party_color, fontsize=20, pad=20, fontweight='bold')
        ax.set_xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
        
        # Add gridlines
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Calculate x-axis padding - ensure bars start at 0
        max_volume = data['TradeAmount'].max() / 1e6
        ax.set_xlim(left=0, right=max_volume * 1.4)  # Increased right padding
        
        # Set background color for consistency
        ax.set_facecolor(COLORS['background'])
        
        return sector_color_map
    
    # Create charts for each party
    dem_sectors = create_party_chart(ax1, dem_stocks, "Democrats", COLORS['democrat'])
    rep_sectors = create_party_chart(ax2, rep_stocks, "Republicans", COLORS['republican'])
    
    # Combine sector colors from both charts for the legend
    all_sectors = {**dem_sectors, **rep_sectors}
    
    # Create nicer legend patches for sectors
    legend_patches = [
        mpatches.Patch(color=color, label=sector, alpha=0.9, linewidth=0) 
        for sector, color in all_sectors.items()
    ]
    
    # Add legend at the bottom with better positioning and styling
    legend = fig.legend(
        handles=legend_patches, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.01),  # Position closer to the plot
        ncol=min(5, len(all_sectors)),
        fontsize=11, 
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor=COLORS['grid'],
        framealpha=0.9
    )
    
    # Add source watermark
    plt.figtext(0.5, -0.02, 'Data source: Quiver Quantitative', fontsize=8, ha='center', alpha=0.7)
    
    # Adjust layout with improved spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Better rect parameters
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'top_stocks_by_party.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved top stocks by party visualization to: {output_path}")
    plt.close()
    
    return output_path


def plot_combined_top_stocks(df, top_n=10):
    """Create a horizontal bar chart of top stocks overall (both parties combined)"""
    print(f"Creating visualization of top {top_n} stocks overall...")
    
    # Group by ticker to get total volume
    ticker_volume = df.groupby('Ticker')['TradeAmount'].sum().reset_index()
    
    # Get ticker sector mapping
    ticker_sector = df.groupby('Ticker')['Sector'].first().to_dict()
    
    # Add sector information
    ticker_volume['Sector'] = ticker_volume['Ticker'].map(ticker_sector)
    
    # Get transaction counts for each ticker
    ticker_counts = df.groupby('Ticker').size().reset_index(name='TradeCount')
    ticker_volume = ticker_volume.merge(ticker_counts, on='Ticker')
    
    # Get top N tickers by volume
    top_tickers = ticker_volume.nlargest(top_n, 'TradeAmount')
    
    # Calculate party breakdown for each top ticker
    party_breakdown = {}
    for ticker in top_tickers['Ticker']:
        ticker_data = df[df['Ticker'] == ticker]
        dem_amount = ticker_data[ticker_data['PartyName'] == 'Democrats']['TradeAmount'].sum()
        rep_amount = ticker_data[ticker_data['PartyName'] == 'Republicans']['TradeAmount'].sum()
        party_breakdown[ticker] = (dem_amount, rep_amount)
    
    # Add party breakdown to top tickers
    top_tickers['DemAmount'] = top_tickers['Ticker'].map(lambda x: party_breakdown[x][0])
    top_tickers['RepAmount'] = top_tickers['Ticker'].map(lambda x: party_breakdown[x][1])
    
    # Sort by total trade amount
    top_tickers = top_tickers.sort_values('TradeAmount', ascending=True)
    
    # Create plot with improved dimensions
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Create positions for bars
    positions = np.arange(len(top_tickers))
    
    # Get sector colors
    sector_color_map = {sector: SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) 
                       for sector in top_tickers['Sector'].unique()}
    
    # Plot stacked horizontal bars for party breakdown with improved colors
    dem_bars = ax.barh(positions, top_tickers['DemAmount'] / 1e6, 
                      height=0.7, color=COLORS['democrat'], alpha=0.9, 
                      label='Democrats')
    rep_bars = ax.barh(positions, top_tickers['RepAmount'] / 1e6, 
                      height=0.7, color=COLORS['republican'], alpha=0.9,
                      left=top_tickers['DemAmount'] / 1e6, label='Republicans')
    
    # Set y-ticks and labels - just ticker names without sector labels
    ax.set_yticks(positions)
    ax.set_yticklabels(top_tickers['Ticker'], fontsize=13, fontweight='bold')
    
    # Add value labels to the right of each bar with improved highlighting
    for i, (amount, count) in enumerate(zip(top_tickers['TradeAmount'], top_tickers['TradeCount'])):
        # Format by thousands or millions
        if amount < 1e6:
            formatted_value = f"${amount/1e3:.1f}K"
        else:
            formatted_value = f"${amount/1e6:.1f}M"
            
        # Add the label with trade count and background highlight
        ax.text(amount/1e6 + 0.5, i, f"{formatted_value} ({count} trades)", 
                ha='left', va='center', color=COLORS['text'], fontsize=11,
                bbox=dict(facecolor=COLORS['background'], alpha=0.7, edgecolor=None, pad=2))
    
    # Add title and labels
    ax.set_title(f'Top {top_n} Most Traded Stocks - 2025 Q1', fontsize=20, pad=20, fontweight='bold')
    ax.set_xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
    
    # Add gridlines
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Calculate x-axis padding
    max_volume = top_tickers['TradeAmount'].max() / 1e6
    ax.set_xlim(left=0, right=max_volume * 1.3)
    
    # Create legend for parties with improved style
    party_patches = [
        mpatches.Patch(color=COLORS['democrat'], label='Democrats', alpha=0.9, linewidth=0),
        mpatches.Patch(color=COLORS['republican'], label='Republicans', alpha=0.9, linewidth=0)
    ]
    
    # Create sector legend patches with better appearance
    sector_patches = [
        mpatches.Patch(color=color, label=sector, alpha=0.9, linewidth=0) 
        for sector, color in sector_color_map.items()
    ]
    
    # Add party legend with better styling
    party_legend = ax.legend(
        handles=party_patches, 
        loc='upper right', 
        fontsize=12, 
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor=COLORS['grid'],
        framealpha=0.9
    )
    
    # Add second legend for sectors with improved positioning and style
    plt.gca().add_artist(party_legend)  # Keep the first legend
    sector_legend = plt.figlegend(
        handles=sector_patches, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.01), 
        ncol=min(4, len(sector_color_map)), 
        fontsize=11, 
        frameon=True, 
        facecolor=COLORS['background'],
        edgecolor=COLORS['grid'],
        framealpha=0.9
    )
    
    # Add source watermark
    plt.figtext(0.5, -0.03, 'Data source: Quiver Quantitative', fontsize=8, ha='center', alpha=0.7)
    
    # Adjust layout with improved spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'top_stocks_overall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved top stocks overall visualization to: {output_path}")
    plt.close()
    
    return output_path


def main():
    """Main function to run the top stocks analysis"""
    print("Starting Top Stocks by Party Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Get top stocks by party
        dem_top, rep_top = get_top_stocks_by_party(df)
        
        # Create visualizations
        plot_top_stocks_by_party(dem_top, rep_top)
        plot_combined_top_stocks(df)
        
        print("Top stocks analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during top stocks analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 