"""
Political Trading Dashboard - 2025 Q1
-------------------------------------

A vibrant, modern dashboard visualizing political trading activity with:
- Glowing neon-style visuals
- Gradient-filled elements
- Modern typography and layouts
- Interactive-looking elements
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# Define vibrant dashboard color scheme
COLORS = {
    'background': '#0A0E17',      # Very dark blue background
    'text': '#FFFFFF',            # White text
    'grid': '#1A1E2E',            # Dark grid lines
    'democrat': '#3366FF',        # Bright blue
    'democrat_glow': '#66A3FF',   # Light blue glow
    'republican': '#FF3366',      # Bright pink/red
    'republican_glow': '#FF66A3', # Light pink glow
    'accent1': '#FF9933',         # Bright orange
    'accent2': '#33CCFF',         # Cyan
    'accent3': '#CC33FF',         # Purple
    'accent4': '#FFCC33',         # Gold
    'panel': '#121824'            # Slightly lighter panel background
}

# Create vibrant sector color palette with neon effect
SECTOR_COLORS = {
    'Information Technology': ('#00FFCC', '#00CCAA'),     # Cyan
    'Health Care': ('#3366FF', '#0044CC'),               # Blue
    'Consumer Discretionary': ('#FF3366', '#CC1444'),     # Pink
    'Financials': ('#FFCC00', '#CC9900'),               # Gold
    'Communication Services': ('#CC33FF', '#9900CC'),     # Purple
    'Industrials': ('#FF9933', '#CC6600'),              # Orange
    'Energy': ('#3399FF', '#0066CC'),                   # Light blue
    'Consumer Staples': ('#00FF66', '#00CC44'),          # Green
    'Materials': ('#FF6633', '#CC4400'),                # Orange-red
    'Real Estate': ('#FF66CC', '#CC33AA'),              # Pink-purple
    'Utilities': ('#33CCCC', '#009999'),                # Teal
    'Unknown': ('#7788AA', '#556688')                   # Gray-blue
}

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
    
    return dem_top, rep_top

def calculate_party_stats(df):
    """Calculate overall party statistics"""
    # Total trading by party
    party_totals = df.groupby('PartyName')['TradeAmount'].sum().reset_index()
    party_counts = df.groupby('PartyName').size().reset_index(name='TotalTrades')
    party_stats = party_totals.merge(party_counts, on='PartyName')
    
    # Convert to dict for easy lookup
    party_data = {}
    for _, row in party_stats.iterrows():
        party_data[row['PartyName']] = {
            'total_amount': row['TradeAmount'],
            'trade_count': row['TotalTrades']
        }
    
    # Add sector breakdown
    sector_by_party = df.groupby(['PartyName', 'Sector'])['TradeAmount'].sum().reset_index()
    
    for party in ['Democrats', 'Republicans']:
        party_sectors = sector_by_party[sector_by_party['PartyName'] == party]
        party_data[party]['sectors'] = {}
        
        # Get top 3 sectors
        top_sectors = party_sectors.nlargest(3, 'TradeAmount')
        total = party_data[party]['total_amount']
        
        for _, row in top_sectors.iterrows():
            party_data[party]['sectors'][row['Sector']] = row['TradeAmount'] / total * 100
            
    return party_data

def setup_dashboard_style():
    """Configure the matplotlib style for a modern dashboard look"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.3,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'legend.facecolor': COLORS['panel'],
        'legend.edgecolor': COLORS['grid'],
        'savefig.facecolor': COLORS['background'],
        'figure.autolayout': False
    })

def apply_glow_effect(artist, glow_color, n_glow_lines=10, glow_size=10):
    """Apply a neon glow effect to a matplotlib artist (text, lines, etc.)"""
    alpha_values = np.linspace(0.5, 0, n_glow_lines)
    glow_lines = []
    
    for alpha in alpha_values:
        effect = path_effects.withStroke(
            linewidth=glow_size, 
            foreground=glow_color, 
            alpha=alpha
        )
        artist.set_path_effects([effect] + artist.get_path_effects())

def create_shiny_gradient_fill(ax, x, y, color_pair, height, alpha=0.9):
    """Create a shiny gradient-filled horizontal bar"""
    main_color, light_color = color_pair
    
    # Create gradient colors
    cmap = LinearSegmentedColormap.from_list('custom_gradient', [main_color, light_color, main_color], N=256)
    
    # Draw the bar with gradient
    n_segments = 50  # More segments = smoother gradient
    segment_width = x / n_segments
    
    for i in range(n_segments):
        segment_x = i * segment_width
        color = cmap(i / n_segments)
        ax.barh(y, segment_width, left=segment_x, height=height, color=color, alpha=alpha, 
               edgecolor=None, zorder=10)
    
    # Add glow
    glow_x = x * 1.02  # Slightly longer to create glow overflow
    glow = ax.barh(y, glow_x, height=height*1.5, left=0, alpha=0.15, 
                  color=light_color, edgecolor=None, zorder=5)
    
    # Add highlight on top
    highlight_height = height * 0.3
    highlight = ax.barh(y + (height-highlight_height)/2, x * 0.95, 
                       height=highlight_height, left=x*0.025, 
                       color='white', alpha=0.2, edgecolor=None, zorder=15)
    
    return glow, highlight

def create_stock_dashboard(dem_stocks, rep_stocks, party_stats):
    """Create a modern, vibrant dashboard of stock trading activity"""
    print("Creating vibrant dashboard visualization...")
    
    # Sort datasets
    dem_stocks = dem_stocks.sort_values('TradeAmount', ascending=True)
    rep_stocks = rep_stocks.sort_values('TradeAmount', ascending=True)
    
    # Convert to millions for cleaner display
    dem_stocks['TradeAmount_M'] = dem_stocks['TradeAmount'] / 1e6
    rep_stocks['TradeAmount_M'] = rep_stocks['TradeAmount'] / 1e6
    
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(24, 14), dpi=100)
    
    # Define grid layout
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[0.5, 3, 3, 0.5])
    
    # Header area
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis('off')
    
    # Main title with glow effect
    title = header_ax.text(0.5, 0.5, 'POLITICAL TRADING DASHBOARD: Q1 2025', 
                         ha='center', va='center', fontsize=26, fontweight='bold',
                         color=COLORS['text'])
    apply_glow_effect(title, '#6688FF', n_glow_lines=15, glow_size=15)
    
    # Subtitle
    header_ax.text(0.5, 0.1, 'Top Stocks & Trading Analysis', 
                 ha='center', va='center', fontsize=14, 
                 color=COLORS['text'], alpha=0.8)
    
    # Add Democrat stocks bar chart
    dem_ax = fig.add_subplot(gs[1, :2])
    dem_ax.set_facecolor(COLORS['panel'])
    
    # Add stylized background panel with rounded corners
    panel_patch = FancyBboxPatch(
        (-0.05, -0.05), 1.1, 1.1, 
        boxstyle=f"round,pad=0.02,rounding_size=0.02", 
        transform=dem_ax.transAxes,
        facecolor=COLORS['panel'],
        edgecolor=COLORS['democrat'],
        linewidth=1.5,
        alpha=0.9,
        zorder=0
    )
    dem_ax.add_patch(panel_patch)
    
    # Democrat title with glow effect
    dem_title = dem_ax.text(0.5, 1.05, 'DEMOCRATS', ha='center', va='bottom', 
                         fontsize=20, fontweight='bold', color=COLORS['democrat'])
    apply_glow_effect(dem_title, COLORS['democrat_glow'], n_glow_lines=10, glow_size=10)
    
    # Add total trading amount for Democrats
    dem_total = f"${party_stats['Democrats']['total_amount']/1e6:.1f}M"
    dem_count = f"{party_stats['Democrats']['trade_count']} trades"
    dem_ax.text(0.97, 0.97, dem_total, ha='right', va='top', 
              fontsize=16, fontweight='bold', color=COLORS['democrat'])
    dem_ax.text(0.97, 0.92, dem_count, ha='right', va='top',
              fontsize=12, color=COLORS['text'], alpha=0.7)
    
    # Draw Democrat bars with gradient fills and glow effects
    bar_height = 0.6
    for i, (ticker, amount, sector) in enumerate(zip(
        dem_stocks['Ticker'], dem_stocks['TradeAmount_M'], dem_stocks['Sector']
    )):
        # Get sector color
        color_pair = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])
        
        # Create gradient filled bar
        create_shiny_gradient_fill(dem_ax, amount, i, color_pair, bar_height)
        
        # Add value label with glow
        if amount < 1:
            formatted_value = f"${amount*1000:.0f}K"
        else:
            formatted_value = f"${amount:.1f}M"
            
        value_label = dem_ax.text(
            amount + 0.3, i, formatted_value,
            ha='left', va='center', fontsize=12, fontweight='bold', color='white'
        )
        apply_glow_effect(value_label, color_pair[0], n_glow_lines=5, glow_size=3)
        
        # Add ticker label
        ticker_label = dem_ax.text(
            -0.5, i, ticker, ha='right', va='center', 
            fontsize=13, fontweight='bold', color='white'
        )
        apply_glow_effect(ticker_label, color_pair[0], n_glow_lines=5, glow_size=3)
    
    # Configure Democrat chart
    dem_ax.set_yticks(range(len(dem_stocks)))
    dem_ax.set_yticklabels([])  # Hide default y labels since we added custom ones
    dem_ax.set_xlabel('Trading Volume ($ Millions)', fontsize=12, labelpad=10)
    dem_ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:.0f}"))
    dem_ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.3)
    dem_ax.spines['top'].set_visible(False)
    dem_ax.spines['right'].set_visible(False)
    dem_ax.spines['bottom'].set_color(COLORS['grid'])
    dem_ax.spines['left'].set_color(COLORS['grid'])
    
    # Set appropriate limits with space for labels
    max_dem_amount = dem_stocks['TradeAmount_M'].max()
    dem_ax.set_xlim(-0.5, max_dem_amount * 1.2)
    
    # Add Republican stocks bar chart
    rep_ax = fig.add_subplot(gs[1, 2:])
    rep_ax.set_facecolor(COLORS['panel'])
    
    # Add stylized background panel with rounded corners
    panel_patch = FancyBboxPatch(
        (-0.05, -0.05), 1.1, 1.1, 
        boxstyle=f"round,pad=0.02,rounding_size=0.02", 
        transform=rep_ax.transAxes,
        facecolor=COLORS['panel'],
        edgecolor=COLORS['republican'],
        linewidth=1.5,
        alpha=0.9,
        zorder=0
    )
    rep_ax.add_patch(panel_patch)
    
    # Republican title with glow effect
    rep_title = rep_ax.text(0.5, 1.05, 'REPUBLICANS', ha='center', va='bottom', 
                         fontsize=20, fontweight='bold', color=COLORS['republican'])
    apply_glow_effect(rep_title, COLORS['republican_glow'], n_glow_lines=10, glow_size=10)
    
    # Add total trading amount for Republicans
    rep_total = f"${party_stats['Republicans']['total_amount']/1e6:.1f}M"
    rep_count = f"{party_stats['Republicans']['trade_count']} trades"
    rep_ax.text(0.97, 0.97, rep_total, ha='right', va='top', 
              fontsize=16, fontweight='bold', color=COLORS['republican'])
    rep_ax.text(0.97, 0.92, rep_count, ha='right', va='top',
              fontsize=12, color=COLORS['text'], alpha=0.7)
    
    # Draw Republican bars with gradient fills and glow effects
    for i, (ticker, amount, sector) in enumerate(zip(
        rep_stocks['Ticker'], rep_stocks['TradeAmount_M'], rep_stocks['Sector']
    )):
        # Get sector color
        color_pair = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])
        
        # Create gradient filled bar
        create_shiny_gradient_fill(rep_ax, amount, i, color_pair, bar_height)
        
        # Add value label with glow
        if amount < 1:
            formatted_value = f"${amount*1000:.0f}K"
        else:
            formatted_value = f"${amount:.1f}M"
            
        value_label = rep_ax.text(
            amount + 0.3, i, formatted_value,
            ha='left', va='center', fontsize=12, fontweight='bold', color='white'
        )
        apply_glow_effect(value_label, color_pair[0], n_glow_lines=5, glow_size=3)
        
        # Add ticker label
        ticker_label = rep_ax.text(
            -0.5, i, ticker, ha='right', va='center', 
            fontsize=13, fontweight='bold', color='white'
        )
        apply_glow_effect(ticker_label, color_pair[0], n_glow_lines=5, glow_size=3)
    
    # Configure Republican chart
    rep_ax.set_yticks(range(len(rep_stocks)))
    rep_ax.set_yticklabels([])  # Hide default y labels
    rep_ax.set_xlabel('Trading Volume ($ Millions)', fontsize=12, labelpad=10)
    rep_ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f"${x:.0f}"))
    rep_ax.grid(axis='x', linestyle='-', linewidth=0.3, alpha=0.3)
    rep_ax.spines['top'].set_visible(False)
    rep_ax.spines['right'].set_visible(False)
    rep_ax.spines['bottom'].set_color(COLORS['grid'])
    rep_ax.spines['left'].set_color(COLORS['grid'])
    
    # Set appropriate limits with space for labels
    max_rep_amount = rep_stocks['TradeAmount_M'].max()
    rep_ax.set_xlim(-0.5, max_rep_amount * 1.2)
    
    # Add sector breakdown donut charts
    dem_sector_ax = fig.add_subplot(gs[2, 0:2])
    dem_sector_ax.set_facecolor(COLORS['panel'])
    
    # Add stylized background panel for Democrat sectors
    panel_patch = FancyBboxPatch(
        (-0.05, -0.05), 1.1, 1.1, 
        boxstyle=f"round,pad=0.02,rounding_size=0.02", 
        transform=dem_sector_ax.transAxes,
        facecolor=COLORS['panel'],
        edgecolor=COLORS['democrat'],
        linewidth=1.5,
        alpha=0.9,
        zorder=0
    )
    dem_sector_ax.add_patch(panel_patch)
    
    # Democrat sectors title with glow
    sectors_title = dem_sector_ax.text(0.5, 1.05, 'TOP SECTORS - DEMOCRATS', 
                                     ha='center', va='bottom', fontsize=16, 
                                     fontweight='bold', color=COLORS['democrat'])
    apply_glow_effect(sectors_title, COLORS['democrat_glow'], n_glow_lines=8, glow_size=8)
    
    # Create Democrat sectors donut chart
    dem_sectors = party_stats['Democrats']['sectors']
    dem_sector_labels = list(dem_sectors.keys())
    dem_sector_values = list(dem_sectors.values())
    dem_sector_colors = [SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])[0] for sector in dem_sector_labels]
    
    # Calculate remaining percentage for "Other"
    other_pct = 100 - sum(dem_sector_values)
    if other_pct > 0:
        dem_sector_labels.append('Others')
        dem_sector_values.append(other_pct)
        dem_sector_colors.append(SECTOR_COLORS['Unknown'][0])
    
    # Draw donut
    dem_sector_ax.pie(
        dem_sector_values, 
        labels=None,
        colors=dem_sector_colors,
        wedgeprops=dict(width=0.4, edgecolor=COLORS['background'], linewidth=2),
        startangle=90,
        counterclock=False
    )
    
    # Add center circle for donut effect with glow
    center_circle = plt.Circle((0, 0), 0.3, fc=COLORS['panel'])
    dem_sector_ax.add_artist(center_circle)
    
    # Add sector labels with percentages and glow effects
    for i, (label, pct, color) in enumerate(zip(dem_sector_labels, dem_sector_values, dem_sector_colors)):
        angle = (sum(dem_sector_values[:i]) + pct/2) * 3.6  # Convert to degrees
        angle_rad = np.radians(90 - angle)  # Adjust angle
        
        # Calculate position
        x = 0.65 * np.cos(angle_rad)
        y = 0.65 * np.sin(angle_rad)
        
        # Label alignment based on position
        ha = 'center'
        if x < -0.3:
            ha = 'right'
        elif x > 0.3:
            ha = 'left'
            
        # Add sector name and percentage
        sector_label = dem_sector_ax.text(
            x, y, f"{label}\n{pct:.1f}%",
            ha=ha, va='center', fontsize=10, fontweight='bold',
            color=color
        )
        apply_glow_effect(sector_label, color, n_glow_lines=5, glow_size=5)
    
    # Add Republican sector breakdown
    rep_sector_ax = fig.add_subplot(gs[2, 2:])
    rep_sector_ax.set_facecolor(COLORS['panel'])
    
    # Add stylized background panel for Republican sectors
    panel_patch = FancyBboxPatch(
        (-0.05, -0.05), 1.1, 1.1, 
        boxstyle=f"round,pad=0.02,rounding_size=0.02", 
        transform=rep_sector_ax.transAxes,
        facecolor=COLORS['panel'],
        edgecolor=COLORS['republican'],
        linewidth=1.5,
        alpha=0.9,
        zorder=0
    )
    rep_sector_ax.add_patch(panel_patch)
    
    # Republican sectors title with glow
    sectors_title = rep_sector_ax.text(0.5, 1.05, 'TOP SECTORS - REPUBLICANS', 
                                     ha='center', va='bottom', fontsize=16, 
                                     fontweight='bold', color=COLORS['republican'])
    apply_glow_effect(sectors_title, COLORS['republican_glow'], n_glow_lines=8, glow_size=8)
    
    # Create Republican sectors donut chart
    rep_sectors = party_stats['Republicans']['sectors']
    rep_sector_labels = list(rep_sectors.keys())
    rep_sector_values = list(rep_sectors.values())
    rep_sector_colors = [SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])[0] for sector in rep_sector_labels]
    
    # Calculate remaining percentage for "Other"
    other_pct = 100 - sum(rep_sector_values)
    if other_pct > 0:
        rep_sector_labels.append('Others')
        rep_sector_values.append(other_pct)
        rep_sector_colors.append(SECTOR_COLORS['Unknown'][0])
    
    # Draw donut
    rep_sector_ax.pie(
        rep_sector_values, 
        labels=None,
        colors=rep_sector_colors,
        wedgeprops=dict(width=0.4, edgecolor=COLORS['background'], linewidth=2),
        startangle=90,
        counterclock=False
    )
    
    # Add center circle for donut effect
    center_circle = plt.Circle((0, 0), 0.3, fc=COLORS['panel'])
    rep_sector_ax.add_artist(center_circle)
    
    # Add sector labels with percentages and glow effects
    for i, (label, pct, color) in enumerate(zip(rep_sector_labels, rep_sector_values, rep_sector_colors)):
        angle = (sum(rep_sector_values[:i]) + pct/2) * 3.6  # Convert to degrees
        angle_rad = np.radians(90 - angle)  # Adjust angle
        
        # Calculate position
        x = 0.65 * np.cos(angle_rad)
        y = 0.65 * np.sin(angle_rad)
        
        # Label alignment based on position
        ha = 'center'
        if x < -0.3:
            ha = 'right'
        elif x > 0.3:
            ha = 'left'
            
        # Add sector name and percentage
        sector_label = rep_sector_ax.text(
            x, y, f"{label}\n{pct:.1f}%",
            ha=ha, va='center', fontsize=10, fontweight='bold',
            color=color
        )
        apply_glow_effect(sector_label, color, n_glow_lines=5, glow_size=5)
    
    # Add legend for sectors
    legend_ax = fig.add_subplot(gs[3, :])
    legend_ax.axis('off')
    
    # Gather all unique sectors
    all_sectors = set()
    for sector_dict in [party_stats['Democrats']['sectors'], party_stats['Republicans']['sectors']]:
        all_sectors.update(sector_dict.keys())
    all_sectors = sorted(all_sectors)
    
    # Create sector color patches for legend
    legend_patches = []
    for sector in all_sectors:
        color = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])[0]
        patch = mpatches.Patch(color=color, label=sector, alpha=0.9)
        legend_patches.append(patch)
    
    # Add legend with glow effect on the text
    legend = legend_ax.legend(
        handles=legend_patches,
        loc='center',
        ncol=min(5, len(all_sectors)),
        fontsize=11,
        frameon=False
    )
    
    # Apply glow to legend text
    for text in legend.get_texts():
        sector = text.get_text()
        color = SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown'])[0]
        apply_glow_effect(text, color, n_glow_lines=3, glow_size=3)
    
    # Add source watermark with subtle glow
    source_text = fig.text(
        0.5, 0.02, 
        'Data source: Quiver Quantitative | Q1 2025 Trading Activity', 
        ha='center', fontsize=9, color=COLORS['text'], alpha=0.6
    )
    apply_glow_effect(source_text, '#6688FF', n_glow_lines=3, glow_size=3)
    
    # Adjust overall layout
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    
    # Save high-resolution output
    output_path = os.path.join(OUTPUT_DIR, 'political_trading_dashboard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved dashboard visualization to: {output_path}")
    plt.close()
    
    return output_path

def main():
    """Main function to run the dashboard creation"""
    print("Starting Modern Dashboard Creation...")
    
    # Setup the dashboard style
    setup_dashboard_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Get top stocks by party
        dem_top, rep_top = get_top_stocks_by_party(df)
        
        # Calculate overall party statistics
        party_stats = calculate_party_stats(df)
        
        # Create modern dashboard visualization
        create_stock_dashboard(dem_top, rep_top, party_stats)
        
        print("Modern dashboard creation complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during dashboard creation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 