"""
Top 5 Stocks Traded by Party - 2025 Q1 (Modern Seaborn Version)
--------------------------------------------------------------

This script uses Seaborn to create modern visualizations of:
1. Top 5 stocks traded by Democrats 
2. Top 5 stocks traded by Republicans
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'Output')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')

# Define custom color scheme
COLORS = {
    'background': '#121212',
    'text': '#FFFFFF',
    'grid': '#333333',
    'democrat': '#3C99DC',
    'republican': '#E91D0E',
}

# Sector-based color mapping
SECTOR_COLORS = {
    'Information Technology': '#58D68D',
    'Health Care': '#3498DB',
    'Consumer Discretionary': '#E74C3C',
    'Financials': '#F4D03F',
    'Communication Services': '#9B59B6',
    'Industrials': '#E67E22',
    'Energy': '#2C3E50',
    'Consumer Staples': '#1ABC9C',
    'Materials': '#D35400',
    'Real Estate': '#CD6155',
    'Utilities': '#7F8C8D',
    'Unknown': '#95A5A6'
}

def setup_plot_style():
    """Setup modern Seaborn plot style"""
    sns.set_theme(style="darkgrid")
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

def plot_top_stocks_by_party_seaborn(dem_stocks, rep_stocks):
    """Create a modern Seaborn visualization of top stocks by party"""
    print("Creating top stocks by party visualization with Seaborn...")
    
    # Sort datasets
    dem_stocks = dem_stocks.sort_values('TradeAmount', ascending=True)
    rep_stocks = rep_stocks.sort_values('TradeAmount', ascending=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
    fig.suptitle('Top 5 Stocks Traded by Party - 2025 Q1', fontsize=22, y=0.98)
    
    # Democrats chart with Seaborn
    dem_colors = [SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) for sector in dem_stocks['Sector']]
    sns.barplot(
        x='TradeAmount', y='Ticker', 
        data=dem_stocks, 
        palette=dem_colors, 
        orient='h',
        ax=ax1,
        alpha=0.9,
        edgecolor='none'
    )
    
    # Add value labels with improved styling
    for i, (amount, count) in enumerate(zip(dem_stocks['TradeAmount'], dem_stocks['TradeCount'])):
        formatted_value = f"${amount/1e6:.1f}M" if amount >= 1e6 else f"${amount/1e3:.1f}K"
        ax1.text(
            amount/1e6 + 0.5, i, 
            f"{formatted_value} ({count} trades)", 
            ha='left', va='center', color=COLORS['text'], fontsize=11,
            bbox=dict(
                facecolor=COLORS['background'], 
                alpha=0.7, 
                edgecolor=None, 
                boxstyle='round,pad=0.5'
            )
        )
    
    # Republicans chart with Seaborn
    rep_colors = [SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']) for sector in rep_stocks['Sector']]
    sns.barplot(
        x='TradeAmount', y='Ticker', 
        data=rep_stocks, 
        palette=rep_colors, 
        orient='h',
        ax=ax2,
        alpha=0.9,
        edgecolor='none'
    )
    
    # Add value labels with improved styling
    for i, (amount, count) in enumerate(zip(rep_stocks['TradeAmount'], rep_stocks['TradeCount'])):
        formatted_value = f"${amount/1e6:.1f}M" if amount >= 1e6 else f"${amount/1e3:.1f}K"
        ax2.text(
            amount/1e6 + 0.5, i, 
            f"{formatted_value} ({count} trades)", 
            ha='left', va='center', color=COLORS['text'], fontsize=11,
            bbox=dict(
                facecolor=COLORS['background'], 
                alpha=0.7, 
                edgecolor=None, 
                boxstyle='round,pad=0.5'
            )
        )
    
    # Styling for Democrats plot
    ax1.set_title("Democrats", color=COLORS['democrat'], fontsize=20, pad=20, fontweight='bold')
    ax1.set_xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='y', labelsize=13, labelcolor=COLORS['text'])
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Convert x-axis to millions
    ax1.set_xticklabels([f"${x/1e6:.0f}M" for x in ax1.get_xticks()])
    
    # Styling for Republicans plot
    ax2.set_title("Republicans", color=COLORS['republican'], fontsize=20, pad=20, fontweight='bold')
    ax2.set_xlabel('Trading Volume ($ Millions)', fontsize=14, labelpad=10)
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='y', labelsize=13, labelcolor=COLORS['text'])
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Convert x-axis to millions
    ax2.set_xticklabels([f"${x/1e6:.0f}M" for x in ax2.get_xticks()])
    
    # Combine sector colors from both charts for the legend
    all_sectors = set(list(dem_stocks['Sector']) + list(rep_stocks['Sector']))
    
    # Create nicer legend patches for sectors
    legend_patches = [
        mpatches.Patch(color=SECTOR_COLORS.get(sector, SECTOR_COLORS['Unknown']), 
                      label=sector, alpha=0.9, linewidth=0) 
        for sector in all_sectors
    ]
    
    # Add legend at the bottom with better positioning and styling
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
    
    # Add source watermark
    plt.figtext(0.5, -0.02, 'Data source: Quiver Quantitative', fontsize=8, ha='center', alpha=0.7)
    
    # Adjust layout with improved spacing
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'top_stocks_by_party_seaborn.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"Saved top stocks visualization to: {output_path}")
    plt.close()
    
    return output_path 