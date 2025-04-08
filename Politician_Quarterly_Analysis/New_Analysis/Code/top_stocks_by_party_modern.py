"""
Top 5 Stocks Traded by Party - 2025 Q1 (Enhanced Modern Version)
---------------------------------------------------------------

This script creates visually refined, modern charts showing:
1. Top 5 stocks traded by Democrats 
2. Top 5 stocks traded by Republicans
3. Sector information with elegant styling
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import patheffects
import matplotlib.ticker as mtick

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# Define custom color scheme - refined dark theme
COLORS = {
    'background': '#0E1117',  # Darker, richer background
    'text': '#F8F9FA',        # Slightly off-white for better readability
    'grid': '#2E3440',        # Subtle grid color
    'democrat': '#4886D0',    # Refined Democratic blue
    'republican': '#E53935',  # Refined Republican red
    'title': '#FFFFFF',       # Pure white for titles
    'subtitle': '#ADB5BD'     # Subtle gray for subtitles
}

# Enhanced sector color palette - more modern with better saturation
SECTOR_COLORS = {
    'Information Technology': '#4ECDC4',  # Mint/Teal
    'Health Care': '#45B7D1',            # Light blue
    'Consumer Discretionary': '#FF6B6B',  # Coral red
    'Financials': '#FFD166',            # Gold
    'Communication Services': '#A78BFA',  # Lavender
    'Industrials': '#F9844A',           # Orange
    'Energy': '#5A6ACF',                # Indigo blue
    'Consumer Staples': '#2BA84A',       # Green
    'Materials': '#F77F00',             # Deep orange
    'Real Estate': '#E76F51',           # Terra cotta
    'Utilities': '#85A392',             # Sage green
    'Unknown': '#9CA3AF'                # Medium gray
}

def load_enriched_data():
    """Load the enriched trading data"""
    input_file = os.path.join(INPUT_DIR, 'enriched_2025q1_trading_data.csv')
    print(f"Loading enriched data from {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Enriched data file not found at {input_file}")
        return None
    
    df = pd.read_csv(input_file)
    
    required_cols = ['Ticker', 'TradeAmount', 'PartyName', 'Sector']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    df['Sector'] = df['Sector'].fillna('Unknown')
    print(f"Loaded {len(df)} trades with ticker information")
    
    return df

def get_top_stocks_by_party(df, top_n=5):
    """Get the top N most traded stocks by volume for each party"""
    print(f"Finding top {top_n} stocks by trading volume for each party...")
    
    party_ticker_volume = df.groupby(['PartyName', 'Ticker'])['TradeAmount'].sum().reset_index()
    ticker_sector = df.groupby('Ticker')['Sector'].first().to_dict()
    party_ticker_volume['Sector'] = party_ticker_volume['Ticker'].map(ticker_sector)
    
    ticker_counts = df.groupby(['PartyName', 'Ticker']).size().reset_index(name='TradeCount')
    party_ticker_volume = party_ticker_volume.merge(ticker_counts, on=['PartyName', 'Ticker'])
    
    democrats = party_ticker_volume[party_ticker_volume['PartyName'] == 'Democrats']
    republicans = party_ticker_volume[party_ticker_volume['PartyName'] == 'Republicans']
    
    dem_top = democrats.nlargest(top_n, 'TradeAmount')
    rep_top = republicans.nlargest(top_n, 'TradeAmount')
    
    return dem_top, rep_top

def setup_modern_style():
    """Setup a modern, elegant plotting style"""
    sns.set_theme(style="dark")
    
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.3,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Roboto', 'Arial', 'Helvetica Neue'],
        'legend.facecolor': COLORS['background'],
        'legend.edgecolor': COLORS['grid'],
        'figure.titlesize': 22,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

def plot_top_stocks_modern(dem_stocks, rep_stocks):
    """Create a modern, visually polished chart of top stocks by party"""
    print("Creating modern top stocks visualization...")
    
    # Sort datasets
    dem_stocks = dem_stocks.sort_values('TradeAmount', ascending=True)
    rep_stocks = rep_stocks.sort_values('TradeAmount', ascending=True)
    
    # Convert to millions for cleaner display
    dem_stocks['TradeAmount_M'] = dem_stocks['TradeAmount'] / 1e6
    rep_stocks['TradeAmount_M'] = rep_stocks['TradeAmount'] / 1e6
    
    # Create figure with modern dimensions and spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), dpi=100)
    plt.subplots_adjust(wspace=0.15)  # Reduce space between subplots
    
    # Add elegant title with subtle highlighting
    title_text = fig.suptitle('Top 5 Stocks Traded by Party - 2025 Q1', 
                             fontsize=24, fontweight='bold', color=COLORS['title'], y=0.98)
    title_text.set_path_effects([
        patheffects.withStroke(linewidth=3, foreground=COLORS['background'])
    ])
    
    # Add subtitle with quarter details
    plt.figtext(0.5, 0.92, 'Trading volume by sector, January - March 2025', 
               ha='center', fontsize=14, color=COLORS['subtitle'])
    
    # Function to create enhanced bar chart
    def create_enhanced_bar_chart(ax, data, party_name, party_color):
        # Create Seaborn barplot with custom colors
        sector_colors = [SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) 
                        for sector in data['Sector']]
        
        # Draw bars with rounded corners - Seaborn doesn't support rounded corners directly
        # so we'll replace it with a custom implementation
        bar_height = 0.65
        for i, (amount, sector) in enumerate(zip(data['TradeAmount_M'], data['Sector'])):
            sector_color = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])
            ax.barh(i, amount, height=bar_height, color=sector_color, alpha=0.9, 
                   edgecolor='none', zorder=10)
            
            # Add inner highlight to bars for 3D effect
            highlight_width = min(amount * 0.98, amount - 0.05)
            if highlight_width > 0:
                ax.barh(i, highlight_width, height=bar_height*0.9, 
                       color=sector_color, alpha=1.0, edgecolor='none', 
                       zorder=11, left=0.05)
        
        # Add elegant value labels with improved styling and formatting
        for i, (amount, count) in enumerate(zip(data['TradeAmount'], data['TradeCount'])):
            # Format values elegantly
            if amount < 1e6:
                formatted_value = f"${amount/1e3:.1f}K"
            else:
                formatted_value = f"${amount/1e6:.1f}M"
            
            # Add label with subtle background glow
            label = ax.text(
                data['TradeAmount_M'].iloc[i] + 0.3, i, 
                f"{formatted_value} ({count} trades)", 
                ha='left', va='center', fontsize=11, fontweight='bold',
                color=COLORS['text'],
                bbox=dict(
                    facecolor=COLORS['background'],
                    edgecolor=None,
                    alpha=0.85,
                    boxstyle='round,pad=0.5'
                )
            )
            label.set_path_effects([
                patheffects.withStroke(linewidth=2, foreground=COLORS['background'])
            ])
        
        # Enhanced styling for y-axis ticker names
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Ticker'], fontweight='bold', fontsize=13)
        
        # Add subtle ticker symbol styling
        for i, ticker_label in enumerate(ax.get_yticklabels()):
            ticker_label.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground=COLORS['background'])
            ])
        
        # Title with party color
        title = ax.set_title(
            party_name, 
            color=party_color, 
            fontsize=20, 
            pad=20, 
            fontweight='bold'
        )
        title.set_path_effects([
            patheffects.withStroke(linewidth=3, foreground=COLORS['background'])
        ])
        
        # Label styling
        ax.set_xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
        
        # Add refined grid lines only on x-axis
        ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.4, zorder=0)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Format x-axis with $ and better tick labels
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:.0f}"))
        
        # Ensure proper limits
        max_volume = data['TradeAmount_M'].max()
        ax.set_xlim(left=0, right=max_volume * 1.2)
        
        return sector_colors
    
    # Create enhanced charts for both parties
    create_enhanced_bar_chart(ax1, dem_stocks, "Democrats", COLORS['democrat'])
    create_enhanced_bar_chart(ax2, rep_stocks, "Republicans", COLORS['republican'])
    
    # Gather unique sectors for legend
    all_sectors = sorted(set(list(dem_stocks['Sector']) + list(rep_stocks['Sector'])))
    
    # Create refined legend patches for sectors
    legend_patches = [
        mpatches.Patch(
            color=SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']), 
            label=sector, 
            alpha=0.9
        ) for sector in all_sectors
    ]
    
    # Add elegant centralized legend
    legend = fig.legend(
        handles=legend_patches, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(all_sectors)),
        fontsize=11, 
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor=COLORS['grid'],
        framealpha=0.9
    )
    
    # Add refined border to legend
    legend.get_frame().set_linewidth(0.5)
    
    # Add subtle source watermark
    source_text = plt.figtext(
        0.5, -0.02, 
        'Data source: Quiver Quantitative', 
        fontsize=9, 
        ha='center', 
        color=COLORS['subtitle'],
        alpha=0.8
    )
    
    # Adjust layout with improved spacing for modern look
    plt.tight_layout(rect=[0, 0.08, 1, 0.9])
    
    # Save figure with high quality
    output_path = os.path.join(OUTPUT_DIR, 'top_stocks_by_party_modern.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'], pad_inches=0.3)
    print(f"Saved modern top stocks visualization to: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Main function to run the modern top stocks analysis"""
    print("Starting Modern Top Stocks Analysis...")
    
    # Setup the modern style
    setup_modern_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Get top stocks by party
        dem_top, rep_top = get_top_stocks_by_party(df)
        
        # Create modern visualization
        plot_top_stocks_modern(dem_top, rep_top)
        
        print("Modern top stocks analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during modern top stocks analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 