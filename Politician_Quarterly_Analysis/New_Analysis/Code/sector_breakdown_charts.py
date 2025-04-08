"""
Sector Breakdown Analysis - 2025 Q1
-----------------------------------

This script creates pie charts showing:
1. Sector breakdown by party (Democrats vs. Republicans)
2. Sector breakdown for buys vs. sells
3. Sector breakdown by party AND transaction type

Charts use a dark theme with clear labeling and consistent styling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap

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
    'buy': '#00FF00',           # Green for buys
    'sell': '#FF0000',          # Red for sells
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
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'legend.facecolor': COLORS['background'],
        'legend.edgecolor': COLORS['grid'],
        'figure.titlesize': 18
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
    required_cols = ['TransactionType', 'TradeAmount', 'PartyName', 'Sector']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    # Remove rows with missing sector data
    df = df.dropna(subset=['Sector'])
    print(f"Loaded {len(df)} trades with sector information")
    
    return df


def prepare_sector_data(df):
    """Prepare sector data for various breakdowns"""
    print("Preparing sector breakdown data...")
    
    # 1. Sector by Party breakdown
    sector_by_party = df.groupby(['PartyName', 'Sector'])['TradeAmount'].sum().reset_index()
    
    # 2. Sector by Transaction Type breakdown
    sector_by_type = df.groupby(['TransactionType', 'Sector'])['TradeAmount'].sum().reset_index()
    
    # 3. Sector by Party AND Transaction Type breakdown
    sector_by_party_type = df.groupby(['PartyName', 'TransactionType', 'Sector'])['TradeAmount'].sum().reset_index()
    
    return sector_by_party, sector_by_type, sector_by_party_type


def plot_sector_by_party(sector_data):
    """Create pie charts showing sector breakdown by party"""
    print("Creating sector breakdown by party pie charts...")
    
    # Create separate dataframes for each party
    dem_data = sector_data[sector_data['PartyName'] == 'Democrats'].copy()
    rep_data = sector_data[sector_data['PartyName'] == 'Republicans'].copy()
    
    # Sort by trade amount
    dem_data = dem_data.sort_values('TradeAmount', ascending=False)
    rep_data = rep_data.sort_values('TradeAmount', ascending=False)
    
    # Keep only top 7 sectors, group others
    def prepare_pie_data(data, n=7):
        if len(data) > n:
            top_sectors = data.head(n).copy()
            others_value = data.iloc[n:]['TradeAmount'].sum()
            others_row = pd.DataFrame({
                'PartyName': [data.iloc[0]['PartyName']],
                'Sector': ['Others'],
                'TradeAmount': [others_value]
            })
            data = pd.concat([top_sectors, others_row], ignore_index=True)
        
        # Calculate percentages
        data['Percentage'] = data['TradeAmount'] / data['TradeAmount'].sum() * 100
        
        return data
    
    dem_pie_data = prepare_pie_data(dem_data)
    rep_pie_data = prepare_pie_data(rep_data)
    
    # Define appealing color palettes
    dem_colors = [
        '#5E9CD3',  # Medium blue
        '#7EB6E9',  # Light blue
        '#3F7FB6',  # Deeper blue
        '#98C2E9',  # Very light blue
        '#4A6F8A',  # Slate blue
        '#30597D',  # Dark blue
        '#76A6CC',  # Sky blue
        '#6B9BC3'   # Dusty blue
    ]
    
    rep_colors = [
        '#D35E5E',  # Medium red
        '#E97E7E',  # Light red
        '#B63F3F',  # Deeper red
        '#E99898',  # Very light red
        '#8A4A4A',  # Brick red
        '#7D3030',  # Dark red
        '#CC7676',  # Dusty rose
        '#C36B6B'   # Rose
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Sector Breakdown by Party - 2025 Q1', fontsize=22, y=0.98)
    
    # Democrat pie chart
    wedges1, texts1, autotexts1 = ax1.pie(
        dem_pie_data['TradeAmount'], 
        labels=dem_pie_data['Sector'],
        autopct='%1.1f%%',
        startangle=90,
        colors=dem_colors[:len(dem_pie_data)],
        wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
    )
    ax1.set_title('Democrats', color='#3C99DC', fontsize=18, pad=20)
    
    # Adjust text properties
    plt.setp(autotexts1, size=10, weight="bold", color='white')
    plt.setp(texts1, size=12)
    
    # Republican pie chart
    wedges2, texts2, autotexts2 = ax2.pie(
        rep_pie_data['TradeAmount'], 
        labels=rep_pie_data['Sector'],
        autopct='%1.1f%%',
        startangle=90,
        colors=rep_colors[:len(rep_pie_data)],
        wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
    )
    ax2.set_title('Republicans', color='#E91D0E', fontsize=18, pad=20)
    
    # Adjust text properties
    plt.setp(autotexts2, size=10, weight="bold", color='white')
    plt.setp(texts2, size=12)
    
    # Add total values to each chart
    total_dem = dem_pie_data['TradeAmount'].sum()
    total_rep = rep_pie_data['TradeAmount'].sum()
    
    dem_value_text = f"Total: ${total_dem/1e6:.1f}M" if total_dem >= 1e6 else f"Total: ${total_dem/1e3:.1f}K"
    rep_value_text = f"Total: ${total_rep/1e6:.1f}M" if total_rep >= 1e6 else f"Total: ${total_rep/1e3:.1f}K"
    
    ax1.text(0, -1.2, dem_value_text, ha='center', fontsize=14, color=COLORS['text'])
    ax2.text(0, -1.2, rep_value_text, ha='center', fontsize=14, color=COLORS['text'])
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'sector_by_party_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved sector by party breakdown to: {output_path}")
    plt.close()


def plot_sector_by_transaction(sector_data):
    """Create pie charts showing sector breakdown by transaction type (buy vs. sell)"""
    print("Creating sector breakdown by transaction type pie charts...")
    
    # Create separate dataframes for buys and sells
    buy_data = sector_data[sector_data['TransactionType'] == 'purchase'].copy()
    sell_data = sector_data[sector_data['TransactionType'] == 'sale'].copy()
    
    # Sort by trade amount
    buy_data = buy_data.sort_values('TradeAmount', ascending=False)
    sell_data = sell_data.sort_values('TradeAmount', ascending=False)
    
    # Keep only top 7 sectors, group others
    def prepare_pie_data(data, n=7):
        if len(data) > n:
            top_sectors = data.head(n).copy()
            others_value = data.iloc[n:]['TradeAmount'].sum()
            others_row = pd.DataFrame({
                'TransactionType': [data.iloc[0]['TransactionType']],
                'Sector': ['Others'],
                'TradeAmount': [others_value]
            })
            data = pd.concat([top_sectors, others_row], ignore_index=True)
        
        # Calculate percentages
        data['Percentage'] = data['TradeAmount'] / data['TradeAmount'].sum() * 100
        
        return data
    
    buy_pie_data = prepare_pie_data(buy_data)
    sell_pie_data = prepare_pie_data(sell_data)
    
    # Define appealing color palettes
    buy_colors = [
        '#58D68D',  # Medium green
        '#82E0AA',  # Light green
        '#2ECC71',  # Brighter green  
        '#ABEBC6',  # Very light green
        '#239B56',  # Forest green
        '#1D8348',  # Dark green
        '#7DCEA0',  # Mint green
        '#73C6B6'   # Sea green
    ]
    
    sell_colors = [
        '#E67E22',  # Orange
        '#EB984E',  # Light orange
        '#D35400',  # Dark orange
        '#F5B041',  # Light amber
        '#F39C12',  # Amber
        '#B9770E',  # Dark amber
        '#EDBB99',  # Peach
        '#DC7633'   # Rust
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Sector Breakdown by Transaction Type - 2025 Q1', fontsize=22, y=0.98)
    
    # Buys pie chart
    wedges1, texts1, autotexts1 = ax1.pie(
        buy_pie_data['TradeAmount'], 
        labels=buy_pie_data['Sector'],
        autopct='%1.1f%%',
        startangle=90,
        colors=buy_colors[:len(buy_pie_data)],
        wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
    )
    ax1.set_title('Purchases', color='#2ECC71', fontsize=18, pad=20)
    
    # Adjust text properties
    plt.setp(autotexts1, size=10, weight="bold", color='white')
    plt.setp(texts1, size=12)
    
    # Sells pie chart
    wedges2, texts2, autotexts2 = ax2.pie(
        sell_pie_data['TradeAmount'], 
        labels=sell_pie_data['Sector'],
        autopct='%1.1f%%',
        startangle=90,
        colors=sell_colors[:len(sell_pie_data)],
        wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
    )
    ax2.set_title('Sales', color='#E67E22', fontsize=18, pad=20)
    
    # Adjust text properties
    plt.setp(autotexts2, size=10, weight="bold", color='white')
    plt.setp(texts2, size=12)
    
    # Add total values to each chart
    total_buy = buy_pie_data['TradeAmount'].sum()
    total_sell = sell_pie_data['TradeAmount'].sum()
    
    buy_value_text = f"Total: ${total_buy/1e6:.1f}M" if total_buy >= 1e6 else f"Total: ${total_buy/1e3:.1f}K"
    sell_value_text = f"Total: ${total_sell/1e6:.1f}M" if total_sell >= 1e6 else f"Total: ${total_sell/1e3:.1f}K"
    
    ax1.text(0, -1.2, buy_value_text, ha='center', fontsize=14, color=COLORS['text'])
    ax2.text(0, -1.2, sell_value_text, ha='center', fontsize=14, color=COLORS['text'])
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'sector_by_transaction_breakdown.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved sector by transaction breakdown to: {output_path}")
    plt.close()


def plot_sector_by_party_and_transaction(sector_data):
    """Create pie charts showing sector breakdown by party AND transaction type"""
    print("Creating sector breakdown by party and transaction type pie charts...")
    
    # Create separate dataframes for each combination
    dem_buy = sector_data[(sector_data['PartyName'] == 'Democrats') & 
                         (sector_data['TransactionType'] == 'purchase')].copy()
    dem_sell = sector_data[(sector_data['PartyName'] == 'Democrats') & 
                          (sector_data['TransactionType'] == 'sale')].copy()
    rep_buy = sector_data[(sector_data['PartyName'] == 'Republicans') & 
                         (sector_data['TransactionType'] == 'purchase')].copy()
    rep_sell = sector_data[(sector_data['PartyName'] == 'Republicans') & 
                          (sector_data['TransactionType'] == 'sale')].copy()
    
    # Sort by trade amount
    dem_buy = dem_buy.sort_values('TradeAmount', ascending=False)
    dem_sell = dem_sell.sort_values('TradeAmount', ascending=False)
    rep_buy = rep_buy.sort_values('TradeAmount', ascending=False)
    rep_sell = rep_sell.sort_values('TradeAmount', ascending=False)
    
    # Keep only top 5 sectors, group others (using fewer sectors since we have 4 charts)
    def prepare_pie_data(data, n=5):
        if len(data) <= 0:
            return None
            
        if len(data) > n:
            top_sectors = data.head(n).copy()
            others_value = data.iloc[n:]['TradeAmount'].sum()
            if others_value > 0:
                others_row = pd.DataFrame({
                    'PartyName': [data.iloc[0]['PartyName']],
                    'TransactionType': [data.iloc[0]['TransactionType']],
                    'Sector': ['Others'],
                    'TradeAmount': [others_value]
                })
                data = pd.concat([top_sectors, others_row], ignore_index=True)
            else:
                data = top_sectors
        
        # Calculate percentages
        data['Percentage'] = data['TradeAmount'] / data['TradeAmount'].sum() * 100
        
        return data
    
    dem_buy_data = prepare_pie_data(dem_buy)
    dem_sell_data = prepare_pie_data(dem_sell)
    rep_buy_data = prepare_pie_data(rep_buy)
    rep_sell_data = prepare_pie_data(rep_sell)
    
    # Define appealing color palettes for each combination
    dem_buy_colors = [
        '#3498DB',  # Medium blue
        '#5DADE2',  # Light blue
        '#2E86C1',  # Deeper blue
        '#85C1E9',  # Sky blue
        '#21618C',  # Dark blue
        '#5499C7'   # Steel blue
    ]
    
    dem_sell_colors = [
        '#7D3C98',  # Medium purple
        '#A569BD',  # Light purple
        '#6C3483',  # Deeper purple
        '#BB8FCE',  # Lavender
        '#4A235A',  # Dark purple
        '#8E44AD'   # Bright purple
    ]
    
    rep_buy_colors = [
        '#E74C3C',  # Medium red
        '#EC7063',  # Light red
        '#CB4335',  # Deeper red
        '#F1948A',  # Dusty rose
        '#943126',  # Dark red
        '#CD6155'   # Rose
    ]
    
    rep_sell_colors = [
        '#D35400',  # Medium orange
        '#E67E22',  # Light orange
        '#BA4A00',  # Deeper orange
        '#EB984E',  # Light amber
        '#873600',  # Dark orange
        '#DC7633'   # Rust
    ]
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Sector Breakdown by Party and Transaction Type - 2025 Q1', 
                fontsize=24, y=0.98)
    
    # Democrat Buys
    if dem_buy_data is not None:
        wedges1, texts1, autotexts1 = ax1.pie(
            dem_buy_data['TradeAmount'], 
            labels=dem_buy_data['Sector'],
            autopct='%1.1f%%',
            startangle=90,
            colors=dem_buy_colors[:len(dem_buy_data)],
            wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
        )
        # Adjust text properties
        plt.setp(autotexts1, size=10, weight="bold", color='white')
        plt.setp(texts1, size=11)
        
        # Add total
        total = dem_buy_data['TradeAmount'].sum()
        value_text = f"Total: ${total/1e6:.1f}M" if total >= 1e6 else f"Total: ${total/1e3:.1f}K"
        ax1.text(0, -1.2, value_text, ha='center', fontsize=12, color=COLORS['text'])
    
    ax1.set_title('Democrat Purchases', color='#3498DB', fontsize=18, pad=20)
    
    # Democrat Sells
    if dem_sell_data is not None:
        wedges2, texts2, autotexts2 = ax2.pie(
            dem_sell_data['TradeAmount'], 
            labels=dem_sell_data['Sector'],
            autopct='%1.1f%%',
            startangle=90,
            colors=dem_sell_colors[:len(dem_sell_data)],
            wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
        )
        # Adjust text properties
        plt.setp(autotexts2, size=10, weight="bold", color='white')
        plt.setp(texts2, size=11)
        
        # Add total
        total = dem_sell_data['TradeAmount'].sum()
        value_text = f"Total: ${total/1e6:.1f}M" if total >= 1e6 else f"Total: ${total/1e3:.1f}K"
        ax2.text(0, -1.2, value_text, ha='center', fontsize=12, color=COLORS['text'])
    
    ax2.set_title('Democrat Sales', color='#7D3C98', fontsize=18, pad=20)
    
    # Republican Buys
    if rep_buy_data is not None:
        wedges3, texts3, autotexts3 = ax3.pie(
            rep_buy_data['TradeAmount'], 
            labels=rep_buy_data['Sector'],
            autopct='%1.1f%%',
            startangle=90,
            colors=rep_buy_colors[:len(rep_buy_data)],
            wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
        )
        # Adjust text properties
        plt.setp(autotexts3, size=10, weight="bold", color='white')
        plt.setp(texts3, size=11)
        
        # Add total
        total = rep_buy_data['TradeAmount'].sum()
        value_text = f"Total: ${total/1e6:.1f}M" if total >= 1e6 else f"Total: ${total/1e3:.1f}K"
        ax3.text(0, -1.2, value_text, ha='center', fontsize=12, color=COLORS['text'])
    
    ax3.set_title('Republican Purchases', color='#E74C3C', fontsize=18, pad=20)
    
    # Republican Sells
    if rep_sell_data is not None:
        wedges4, texts4, autotexts4 = ax4.pie(
            rep_sell_data['TradeAmount'], 
            labels=rep_sell_data['Sector'],
            autopct='%1.1f%%',
            startangle=90,
            colors=rep_sell_colors[:len(rep_sell_data)],
            wedgeprops={'edgecolor': 'w', 'linewidth': 0.5, 'alpha': 0.9}
        )
        # Adjust text properties
        plt.setp(autotexts4, size=10, weight="bold", color='white')
        plt.setp(texts4, size=11)
        
        # Add total
        total = rep_sell_data['TradeAmount'].sum()
        value_text = f"Total: ${total/1e6:.1f}M" if total >= 1e6 else f"Total: ${total/1e3:.1f}K"
        ax4.text(0, -1.2, value_text, ha='center', fontsize=12, color=COLORS['text'])
    
    ax4.set_title('Republican Sales', color='#D35400', fontsize=18, pad=20)
    
    # Add source watermark
    plt.figtext(0.5, 0.01, 'Data source: Quiver Quantitative', fontsize=8, ha='center')
    
    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(OUTPUT_DIR, 'sector_by_party_and_transaction.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved sector by party and transaction breakdown to: {output_path}")
    plt.close()


def main():
    """Main function to run the sector breakdown analysis"""
    print("Starting Sector Breakdown Analysis...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Prepare sector data
        sector_by_party, sector_by_type, sector_by_party_type = prepare_sector_data(df)
        
        # Create pie charts
        plot_sector_by_party(sector_by_party)
        plot_sector_by_transaction(sector_by_type)
        plot_sector_by_party_and_transaction(sector_by_party_type)
        
        print("Sector breakdown analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during sector breakdown analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 