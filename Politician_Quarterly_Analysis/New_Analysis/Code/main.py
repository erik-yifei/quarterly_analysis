"""
Political Trading Data Enrichment - 2025 Q1
-------------------------------------------

This script processes politician trading data for Q1 2025 and enriches it with 
sector/industry information from yfinance API.

The script focuses on:
1. Preprocessing the raw data
2. Fetching sector and industry information for each ticker
3. Saving the enriched dataset for later analysis and visualization
"""

import os
import pandas as pd
import numpy as np
import re
import yfinance as yf
from tqdm import tqdm
import time

# Define constants
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Input')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Output')


def setup_directories():
    """Create input/output directories if they don't exist"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return INPUT_DIR, OUTPUT_DIR


def parse_range_values(range_str):
    """
    Parse the 'Range' column to extract the upper dollar amount.
    Example: "$1,001 - $15,000" -> 15000
    """
    if pd.isna(range_str):
        return 0
    
    # Extract numbers from the range string
    numbers = re.findall(r'[$]([0-9,]+)', str(range_str))
    if not numbers or len(numbers) < 2:
        return 0
    
    # Take the higher number (second value in the range)
    try:
        # Remove commas and convert to float
        return float(numbers[1].replace(',', ''))
    except (ValueError, IndexError):
        return 0


def preprocess_data(file_path):
    """
    Preprocess raw politician trading data
    """
    print("Loading and preprocessing trading data...")
    
    try:
        # Load data and print columns for debugging
        df = pd.read_csv(file_path)
        print(f"Available columns: {df.columns.tolist()}")
        
        # Convert dates to datetime
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        
        # Check if we need to filter by quarter
        if 'Quarter' not in df.columns:
            # Filter for Q1 2025 only (Jan 1 - Mar 31)
            q1_start = pd.Timestamp('2025-01-01')
            q1_end = pd.Timestamp('2025-03-31')
            df = df[(df['TransactionDate'] >= q1_start) & (df['TransactionDate'] <= q1_end)]
            df['Quarter'] = '2025-Q1'
        
        # Clean transaction type if not already present
        if 'TransactionType' not in df.columns:
            def clean_transaction_type(transaction):
                transaction = str(transaction).lower()
                if 'exchange' in transaction:
                    return 'exchange'
                elif any(sale_type in transaction for sale_type in ['sale', 'sold']):
                    return 'sale'
                elif any(purchase_type in transaction for purchase_type in ['purchase', 'bought']):
                    return 'purchase'
                else:
                    return 'other'
            
            df['TransactionType'] = df['Transaction'].apply(clean_transaction_type)
        
        # Filter out exchanges and other transaction types
        df = df[df['TransactionType'].isin(['sale', 'purchase'])]
        
        # Calculate trade amount if not already present
        if 'TradeAmount' not in df.columns and 'Range' in df.columns:
            df['TradeAmount'] = df['Range'].apply(parse_range_values)
        
        # Add party name if not already present
        if 'PartyName' not in df.columns and 'Party' in df.columns:
            df['PartyName'] = df['Party'].map({'D': 'Democrats', 'R': 'Republicans'})
        
        # Print summary of the data
        print("\nData Summary:")
        print(f"Total trades: {len(df)}")
        if 'TransactionDate' in df.columns:
            print(f"Date range: {df['TransactionDate'].min()} to {df['TransactionDate'].max()}")
        print(f"\nTrades by transaction type:")
        print(df['TransactionType'].value_counts())
        print(f"\nTrades by party:")
        print(df['PartyName'].value_counts())
        print(f"\nTotal trade amount: ${df['TradeAmount'].sum():,.2f}")
        
        return df
    
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_stock_info(ticker):
    """
    Get sector and industry information for a stock from yfinance API
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.2)  # Increased delay to avoid rate limits
        
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Print status for each ticker for tracking progress
        print(f"Ticker: {ticker}, Sector: {sector}, Industry: {industry}")
        
        return {
            'Sector': sector,
            'Industry': industry
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return {
            'Sector': 'Unknown',
            'Industry': 'Unknown'
        }


def enrich_with_sectors(df):
    """
    Enrich trading data with sector and industry information
    """
    print("Enriching data with sector information...")
    
    # Get unique tickers
    unique_tickers = df['Ticker'].unique()
    print(f"Found {len(unique_tickers)} unique tickers")
    
    # Create a dictionary to store stock info
    stock_info = {}
    
    # Fetch stock info for each ticker with progress bar
    print("Fetching sector data from yfinance...")
    for ticker in tqdm(unique_tickers):
        stock_info[ticker] = get_stock_info(ticker)
    
    # Add sector and industry columns
    df['Sector'] = df['Ticker'].map(lambda x: stock_info[x]['Sector'])
    df['Industry'] = df['Ticker'].map(lambda x: stock_info[x]['Industry'])
    
    # Replace 'Unknown' with NaN for easier filtering
    df['Sector'].replace('Unknown', np.nan, inplace=True)
    df['Industry'].replace('Unknown', np.nan, inplace=True)
    
    # Print sector distribution
    print("\nSector distribution:")
    print(df['Sector'].value_counts(dropna=False))
    print("\nIndustry distribution (top 10):")
    print(df['Industry'].value_counts(dropna=False).head(10))
    
    # Print summary of enriched data
    print(f"\nPercentage of trades with sector information: {df['Sector'].notna().mean()*100:.1f}%")
    print(f"Percentage of trades with industry information: {df['Industry'].notna().mean()*100:.1f}%")
    
    return df


def save_enriched_data(df):
    """Save the enriched data to CSV files"""
    print("Saving enriched data...")
    
    # Save full dataset
    output_file = os.path.join(OUTPUT_DIR, 'enriched_2025q1_trading_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved enriched data to: {output_file}")
    
    # Save sector summary
    sector_summary = df.groupby('Sector').agg({
        'TradeAmount': 'sum',
        'Representative': 'nunique',
        'Ticker': 'nunique'
    }).reset_index()
    sector_summary.columns = ['Sector', 'TotalInvestment', 'UniquePoliticians', 'UniqueStocks']
    sector_summary = sector_summary.sort_values('TotalInvestment', ascending=False)
    
    summary_file = os.path.join(OUTPUT_DIR, 'sector_summary_2025q1.csv')
    sector_summary.to_csv(summary_file, index=False)
    print(f"Saved sector summary to: {summary_file}")
    
    # Save industry summary
    industry_summary = df.groupby('Industry').agg({
        'TradeAmount': 'sum',
        'Representative': 'nunique',
        'Ticker': 'nunique'
    }).reset_index()
    industry_summary.columns = ['Industry', 'TotalInvestment', 'UniquePoliticians', 'UniqueStocks']
    industry_summary = industry_summary.sort_values('TotalInvestment', ascending=False)
    
    industry_file = os.path.join(OUTPUT_DIR, 'industry_summary_2025q1.csv')
    industry_summary.to_csv(industry_file, index=False)
    print(f"Saved industry summary to: {industry_file}")


def main():
    """Main function to process data and add sector information"""
    print("Starting 2025 Q1 Political Trading Data Enrichment...")
    
    # Setup directories
    setup_directories()
    
    try:
        # Step 1: Preprocess raw data
        input_file = os.path.join(INPUT_DIR, 'cleaned_2025q1_politician_trading_data.csv')
        df = preprocess_data(input_file)
        
        if df is None or df.empty:
            print("Error: No valid data available for analysis.")
            return
        
        # Step 2: Enrich with sector data from yfinance
        enriched_df = enrich_with_sectors(df)
        
        # Step 3: Save enriched data
        save_enriched_data(enriched_df)
        
        print("Data enrichment complete. Enriched data saved to the Output directory.")
        
    except Exception as e:
        print(f"Error during data enrichment: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
