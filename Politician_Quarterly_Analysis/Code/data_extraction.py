"""
Data extraction script for congressional trading data from Quiver Quantitative API.
Fetches ALL available congressional trading data from the API and exports it to a CSV file.
No date filtering is applied - this will retrieve the complete dataset available through your API key.

Note: This extraction script pulls all available data. The analysis scripts will filter for:
- 2024 Q4 (October 1, 2024 - December 31, 2024)
- 2025 Q1 (January 1, 2025 - present)

Data format from Quiver API:
{
    "Representative": "Josh Gottheimer",       # Name of the politician
    "BioGuideID": "G000583",                   # Unique identifier for the politician
    "ReportDate": "2025-02-10",                # Date when the trade was reported
    "TransactionDate": "2025-01-21",           # Date when the trade occurred
    "Ticker": "ORCL",                          # Stock ticker symbol
    "Transaction": "Sale",                     # Type of transaction (Purchase, Sale, Exchange)
    "Range": "$1,001 - $15,000",               # Reported dollar range of the transaction
    "House": "Representatives",                # Chamber (Representatives or Senate)
    "Amount": "1001.0",                        # Numeric value for the transaction
    "Party": "D",                              # Political party (D=Democrat, R=Republican, I=Independent)
    "last_modified": "2025-02-11",             # Last modified date in Quiver's system
    "TickerType": "ST",                        # Type of asset (ST=Stock, etc.)
    "Description": null,                       # Additional description if available
    "ExcessReturn": -5.22254248825401,         # Performance of the stock relative to the market
    "PriceChange": -12.5630179057774,          # Percentage change in the stock price
    "SPYChange": -7.34047541752336             # Percentage change in the S&P 500
}
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
import quiverquant
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
QUIVER_API_KEY = os.getenv("API_KEY")

def setup_directories():
    """Ensures the Input directory exists."""
    input_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Input")
    os.makedirs(input_dir, exist_ok=True)
    return input_dir

def get_congress_trading_data():
    """
    Fetches ALL available congressional trading data from Quiver Quantitative API.
    No date filtering is applied - this retrieves the complete dataset accessible through your API key.
    
    Returns:
        pandas.DataFrame: DataFrame containing all available congressional trading data
    """
    try:
        # Initialize the Quiver client with API key
        print(f"Initializing Quiver client with API key: {QUIVER_API_KEY[:5]}..." + "*" * 10)
        quiver = quiverquant.quiver(QUIVER_API_KEY)
        
        # Fetch trading data - get COMPLETE dataset with no filtering
        print(f"Fetching ALL available congressional trading data (complete dataset)...")
        try:
            # Try to get the data
            congress_data = quiver.congress_trading()
            
            # Check what type of object is returned
            print(f"API returned a {type(congress_data)} object")
            
            # If it's not a DataFrame, try to convert or debug
            if not isinstance(congress_data, pd.DataFrame):
                if isinstance(congress_data, dict) or isinstance(congress_data, list):
                    # If it's a dict or list, try to convert to DataFrame
                    print("Converting to DataFrame...")
                    if isinstance(congress_data, dict):
                        # If it's a dictionary, convert to dataframe
                        congress_data = pd.DataFrame([congress_data])
                    else:
                        # If it's a list, convert to dataframe
                        congress_data = pd.DataFrame(congress_data)
                else:
                    # Otherwise, just print the data for debugging
                    print(f"Unexpected response from API: {congress_data}")
                    return None
            
            # Print the first few rows to see what we got
            print("\nSample data (first 3 rows):")
            print(congress_data.head(3))
            
            # Print the columns to see what fields are available
            print("\nAvailable columns:")
            print(congress_data.columns.tolist())
            
            # Print the date range in the data
            if 'TransactionDate' in congress_data.columns:
                congress_data['Date'] = pd.to_datetime(congress_data['TransactionDate'])
                min_date = congress_data['Date'].min().strftime('%Y-%m-%d')
                max_date = congress_data['Date'].max().strftime('%Y-%m-%d')
                print(f"\nData covers trades from {min_date} to {max_date}")
                print(f"Total records: {len(congress_data)}")
            
        except Exception as e:
            print(f"Error fetching congress trading data: {str(e)}")
            return None
        
        # Continue with processing
        if congress_data.empty:
            print("No trading data found.")
            return None
            
        print(f"Retrieved {len(congress_data)} trading records.")
        
        # Add some useful derived fields (with error handling)
        try:
            if 'Amount' in congress_data.columns:
                congress_data['TradeValue'] = congress_data['Amount'].astype(float)
            
            if 'ReportDate' in congress_data.columns and 'TransactionDate' in congress_data.columns:
                congress_data['DaysToReport'] = (pd.to_datetime(congress_data['ReportDate']) - 
                                              pd.to_datetime(congress_data['TransactionDate'])).dt.days
        except Exception as e:
            print(f"Warning: Could not process some derived fields: {str(e)}")
        
        # Select and order columns for the output
        columns_order = [
            'Representative', 'Party', 'House', 'BioGuideID',
            'TransactionDate', 'ReportDate', 'DaysToReport',
            'Ticker', 'Transaction', 'Range', 'TradeValue',
            'TickerType', 'Description',
            'ExcessReturn', 'PriceChange', 'SPYChange',
            'last_modified'
        ]
        
        # Keep only the columns that exist in the data
        columns_to_keep = [col for col in columns_order if col in congress_data.columns]
        
        return congress_data[columns_to_keep]
    
    except Exception as e:
        print(f"Error fetching data from Quiver Quantitative API: {str(e)}")
        return None

def generate_summary(data):
    """
    Generates a summary of the trading data.
    
    Args:
        data (pandas.DataFrame): Trading data
    
    Returns:
        str: Summary text
    """
    if data is None or data.empty:
        return "No data available for summary."
    
    summary = []
    summary.append(f"Total trades: {len(data)}")
    
    # Date range information
    if 'TransactionDate' in data.columns:
        data['Date'] = pd.to_datetime(data['TransactionDate'])
        min_date = data['Date'].min().strftime('%Y-%m-%d')
        max_date = data['Date'].max().strftime('%Y-%m-%d')
        summary.append(f"\nDate range: {min_date} to {max_date}")
    
    # Count by transaction type (with error handling)
    if 'Transaction' in data.columns:
        transaction_counts = data['Transaction'].value_counts()
        summary.append("\nTransactions by type:")
        for tx_type, count in transaction_counts.items():
            summary.append(f"  {tx_type}: {count}")
    
    # Count by party (with error handling)
    if 'Party' in data.columns:
        party_counts = data['Party'].value_counts()
        summary.append("\nTransactions by party:")
        for party, count in party_counts.items():
            party_name = "Democrat" if party == "D" else "Republican" if party == "R" else "Independent" if party == "I" else party
            summary.append(f"  {party_name}: {count}")
    
    # Most active politicians (with error handling)
    if 'Representative' in data.columns:
        politician_counts = data.groupby('Representative').size().sort_values(ascending=False).head(10)
        summary.append("\nMost active politicians:")
        for politician, count in politician_counts.items():
            summary.append(f"  {politician}: {count} trades")
    
    # Most traded tickers (with error handling)
    if 'Ticker' in data.columns:
        ticker_counts = data.groupby('Ticker').size().sort_values(ascending=False).head(10)
        summary.append("\nMost traded tickers:")
        for ticker, count in ticker_counts.items():
            summary.append(f"  {ticker}: {count} trades")
    
    return "\n".join(summary)

def main():
    """Main function to fetch and export congressional trading data."""
    # Setup directories
    input_dir = setup_directories()
    
    # Check if API key is set
    if not QUIVER_API_KEY:
        print("Error: API key not set. Please create a .env file with your API_KEY=your_api_key_here")
        print("You can sign up for a Quiver Quantitative API key at https://www.quiverquant.com/")
        sys.exit(1)
    
    # Get trading data - ALL available data with no date filtering
    trading_data = get_congress_trading_data()
    
    if trading_data is not None:
        # Get the date range for the filename
        if 'TransactionDate' in trading_data.columns:
            trading_data['Date'] = pd.to_datetime(trading_data['TransactionDate'])
            min_year_month = trading_data['Date'].min().strftime('%Y_%m')
            max_year_month = trading_data['Date'].max().strftime('%Y_%m')
            filename = f"politician_trades_{min_year_month}_to_{max_year_month}.csv"
        else:
            filename = "politician_trades_complete.csv"
            
        # Define output file path
        output_file = os.path.join(input_dir, filename)
        
        # Export to CSV
        trading_data.to_csv(output_file, index=False)
        print(f"Data successfully exported to {output_file}")
        
        # Generate and display summary
        summary = generate_summary(trading_data)
        print("\nSummary of Congressional Trading Data:")
        print(summary)
    else:
        print("Failed to fetch trading data. Please check your API key and try again.")

if __name__ == "__main__":
    main()
