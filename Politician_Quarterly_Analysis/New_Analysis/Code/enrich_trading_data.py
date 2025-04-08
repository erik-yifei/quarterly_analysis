"""
Enrich Trading Data Script

This script takes the cleaned 2025 Q1 trading data and enriches it with additional
information from the yfinance API, including:
- Sector data
- Industry data
- Market cap
- Company name
- 52-week high/low
"""

import pandas as pd
import yfinance as yf
import os
from tqdm import tqdm
import time

def setup_directories():
    """Create input/output directories if they don't exist"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_dir = os.path.join(base_dir, 'Input')
    output_dir = os.path.join(base_dir, 'Output')
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return input_dir, output_dir

def get_stock_info(ticker):
    """
    Get stock information from yfinance API
    Returns a dictionary with relevant information
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.1)
        
        # Get stock info
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown'),
            'MarketCap': info.get('marketCap', 0),
            'CompanyName': info.get('longName', ticker),
            '52WeekHigh': info.get('fiftyTwoWeekHigh', 0),
            '52WeekLow': info.get('fiftyTwoWeekLow', 0),
            'Beta': info.get('beta', 0),
            'PERatio': info.get('forwardPE', 0),
            'DividendYield': info.get('dividendYield', 0),
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return {
            'Sector': 'Unknown',
            'Industry': 'Unknown',
            'MarketCap': 0,
            'CompanyName': ticker,
            '52WeekHigh': 0,
            '52WeekLow': 0,
            'Beta': 0,
            'PERatio': 0,
            'DividendYield': 0,
        }

def enrich_trading_data(input_file):
    """
    Enrich trading data with additional information from yfinance
    """
    print("Loading trading data...")
    df = pd.read_csv(input_file)
    
    # Get unique tickers
    unique_tickers = df['Ticker'].unique()
    print(f"Found {len(unique_tickers)} unique tickers")
    
    # Create a dictionary to store stock info
    stock_info = {}
    
    # Fetch stock info for each ticker with progress bar
    print("Fetching stock information...")
    for ticker in tqdm(unique_tickers):
        stock_info[ticker] = get_stock_info(ticker)
    
    # Create new columns from stock info
    print("Enriching data with stock information...")
    df['Sector'] = df['Ticker'].map(lambda x: stock_info[x]['Sector'])
    df['Industry'] = df['Ticker'].map(lambda x: stock_info[x]['Industry'])
    df['MarketCap'] = df['Ticker'].map(lambda x: stock_info[x]['MarketCap'])
    df['CompanyName'] = df['Ticker'].map(lambda x: stock_info[x]['CompanyName'])
    df['52WeekHigh'] = df['Ticker'].map(lambda x: stock_info[x]['52WeekHigh'])
    df['52WeekLow'] = df['Ticker'].map(lambda x: stock_info[x]['52WeekLow'])
    df['Beta'] = df['Ticker'].map(lambda x: stock_info[x]['Beta'])
    df['PERatio'] = df['Ticker'].map(lambda x: stock_info[x]['PERatio'])
    df['DividendYield'] = df['Ticker'].map(lambda x: stock_info[x]['DividendYield'])
    
    # Calculate market cap category
    def get_market_cap_category(cap):
        if cap >= 200e9:
            return 'Mega Cap'
        elif cap >= 10e9:
            return 'Large Cap'
        elif cap >= 2e9:
            return 'Mid Cap'
        elif cap >= 300e6:
            return 'Small Cap'
        else:
            return 'Micro Cap'
    
    df['MarketCapCategory'] = df['MarketCap'].apply(get_market_cap_category)
    
    return df

def main():
    """Main function to run the enrichment process"""
    print("Starting data enrichment process...")
    
    # Setup directories
    input_dir, output_dir = setup_directories()
    
    # Input file path
    input_file = os.path.join(input_dir, 'cleaned_2025q1_politician_trading_data.csv')
    
    try:
        # Enrich the data
        enriched_df = enrich_trading_data(input_file)
        
        # Save enriched data
        output_file = os.path.join(output_dir, 'enriched_2025q1_trading_data.csv')
        enriched_df.to_csv(output_file, index=False)
        print(f"Saved enriched data to: {output_file}")
        
        # Print summary statistics
        print("\nEnrichment Summary:")
        print(f"Total trades: {len(enriched_df)}")
        print("\nSector distribution:")
        print(enriched_df['Sector'].value_counts())
        print("\nMarket Cap distribution:")
        print(enriched_df['MarketCapCategory'].value_counts())
        
    except Exception as e:
        print(f"Error during enrichment process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 