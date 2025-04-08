"""
Buy vs. Sell Analysis by Party - 2025 Q1
----------------------------------------

This script analyzes the enriched 2025 Q1 politician trading data to compare 
buying and selling behavior between political parties.

Analysis includes:
1. Total buy vs. sell volume by party
2. Buy-sell ratio by party
3. Sector breakdown of buy vs. sell activity
4. Top politicians buy vs. sell behavior
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')  # Use the enriched data from Output dir
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
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif'
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
    required_cols = ['TransactionType', 'TradeAmount', 'PartyName']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return None
    
    print(f"Loaded {len(df)} trades")
    return df


def analyze_buy_sell_volume(df):
    """Analyze buy vs. sell volume by party"""
    print("Analyzing buy vs. sell volume by party...")
    
    # Group by Party and Transaction Type
    party_transaction = df.groupby(['PartyName', 'TransactionType'])['TradeAmount'].sum().reset_index()
    
    # Reshape for easier analysis
    party_volume = party_transaction.pivot(
        index='PartyName', 
        columns='TransactionType', 
        values='TradeAmount'
    ).reset_index()
    
    # Calculate total volume and buy-sell ratio
    party_volume['TotalVolume'] = party_volume['purchase'] + party_volume['sale']
    party_volume['BuySellRatio'] = party_volume['purchase'] / party_volume['sale']
    
    # Rename columns for clarity
    party_volume.columns.name = None
    party_volume = party_volume.rename(columns={
        'purchase': 'BuyVolume',
        'sale': 'SellVolume'
    })
    
    print("\nParty Buy vs. Sell Summary:")
    print(party_volume)
    
    return party_volume


def plot_buy_sell_by_party(volume_data):
    """Create a visualization of buy vs. sell volume by party"""
    print("Creating buy vs. sell visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create positions for grouped bars
    parties = volume_data['PartyName']
    x = np.arange(len(parties))
    width = 0.35
    
    # Create grouped bars
    ax.bar(x - width/2, volume_data['BuyVolume'] / 1e6, width, 
           label='Buy Volume', color=COLORS['buy'], alpha=0.8)
    ax.bar(x + width/2, volume_data['SellVolume'] / 1e6, width, 
           label='Sell Volume', color=COLORS['sell'], alpha=0.8)
    
    # Add data labels
    for i, v in enumerate(volume_data['BuyVolume']):
        ax.text(i - width/2, v/1e6 + 1, f'${v/1e6:.1f}M', 
                ha='center', va='bottom', color=COLORS['text'])
    
    for i, v in enumerate(volume_data['SellVolume']):
        ax.text(i + width/2, v/1e6 + 1, f'${v/1e6:.1f}M', 
                ha='center', va='bottom', color=COLORS['text'])
    
    # Add buy-sell ratio
    for i, ratio in enumerate(volume_data['BuySellRatio']):
        ax.text(i, 0.5, f'Buy/Sell Ratio: {ratio:.2f}', 
                ha='center', va='bottom', fontweight='bold', color=COLORS['text'],
                bbox=dict(facecolor=COLORS['background'], edgecolor=COLORS['grid'], 
                         boxstyle='round,pad=0.5'))
    
    # Add labels and formatting
    ax.set_title('Buy vs. Sell Volume by Party - 2025 Q1', fontsize=18, pad=20)
    ax.set_xlabel('Party', fontsize=14)
    ax.set_ylabel('Volume ($ Millions)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(parties)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'buy_sell_volume_by_party.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved buy vs. sell visualization to: {output_path}")
    plt.close()


def analyze_sector_buy_sell(df):
    """Analyze buy vs. sell activity by sector and party"""
    print("Analyzing buy vs. sell activity by sector and party...")
    
    # Drop trades with unknown sectors
    df_with_sector = df.dropna(subset=['Sector']).copy()
    
    # Group by Sector, Party, and Transaction Type
    sector_party_tx = df_with_sector.groupby(
        ['Sector', 'PartyName', 'TransactionType']
    )['TradeAmount'].sum().reset_index()
    
    # Create a pivot table for buy and sell volume by sector and party
    sector_party_volume = sector_party_tx.pivot_table(
        index=['Sector', 'PartyName'],
        columns='TransactionType',
        values='TradeAmount',
        fill_value=0
    ).reset_index()
    
    # Calculate total volume and net purchases (buy - sell)
    sector_party_volume['TotalVolume'] = sector_party_volume['purchase'] + sector_party_volume['sale']
    sector_party_volume['NetPurchases'] = sector_party_volume['purchase'] - sector_party_volume['sale']
    
    # Rename columns for clarity
    sector_party_volume.columns.name = None
    sector_party_volume = sector_party_volume.rename(columns={
        'purchase': 'BuyVolume',
        'sale': 'SellVolume'
    })
    
    # Sort by total volume
    sector_party_volume = sector_party_volume.sort_values('TotalVolume', ascending=False)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'sector_buy_sell_by_party.csv')
    sector_party_volume.to_csv(output_path, index=False)
    print(f"Saved sector buy vs. sell analysis to: {output_path}")
    
    # Return top sectors by volume
    top_sectors = sector_party_volume.head(10)
    print("\nTop 10 Sectors by Trading Volume:")
    print(top_sectors[['Sector', 'PartyName', 'BuyVolume', 'SellVolume', 'NetPurchases']])
    
    return sector_party_volume


def plot_top_politicians_buy_sell(df, top_n=10):
    """Plot top politicians by buy and sell volume"""
    print(f"Analyzing top {top_n} politicians by trading volume...")
    
    # Group by politician and transaction type
    politician_tx = df.groupby(['Representative', 'PartyName', 'TransactionType'])['TradeAmount'].sum().reset_index()
    
    # Create a pivot table for buy and sell volume by politician
    politician_volume = politician_tx.pivot_table(
        index=['Representative', 'PartyName'],
        columns='TransactionType',
        values='TradeAmount',
        fill_value=0
    ).reset_index()
    
    # Calculate total volume
    politician_volume['TotalVolume'] = politician_volume['purchase'] + politician_volume['sale']
    
    # Calculate net purchases (buy - sell)
    politician_volume['NetPurchases'] = politician_volume['purchase'] - politician_volume['sale']
    
    # Rename columns for clarity
    politician_volume.columns.name = None
    politician_volume = politician_volume.rename(columns={
        'purchase': 'BuyVolume',
        'sale': 'SellVolume'
    })
    
    # Sort by total volume and get top N
    top_politicians = politician_volume.sort_values('TotalVolume', ascending=False).head(top_n)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'politician_buy_sell_volume.csv')
    politician_volume.to_csv(output_path, index=False)
    print(f"Saved all politician buy vs. sell data to: {output_path}")
    
    # Print summary
    print(f"\nTop {top_n} Politicians by Trading Volume:")
    print(top_politicians[['Representative', 'PartyName', 'BuyVolume', 'SellVolume', 'NetPurchases']])
    
    return top_politicians


def main():
    """Main function to run the analysis"""
    print("Starting Buy vs. Sell Analysis by Party for 2025 Q1...")
    
    # Setup plotting style
    setup_plot_style()
    
    try:
        # Load enriched data
        df = load_enriched_data()
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Analyze buy vs. sell volume by party
        party_volume = analyze_buy_sell_volume(df)
        
        # Create visualization
        plot_buy_sell_by_party(party_volume)
        
        # Analyze by sector
        analyze_sector_buy_sell(df)
        
        # Analyze top politicians
        plot_top_politicians_buy_sell(df)
        
        print("Buy vs. Sell analysis complete. Results saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 