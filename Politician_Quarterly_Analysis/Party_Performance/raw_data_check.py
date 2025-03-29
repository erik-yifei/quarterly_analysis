import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf
import time  # Add this import
import json

# Load environment variables
load_dotenv()
QUIVER_API_KEY = os.getenv("API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Add this to your .env file

# Set up the input directory path
INPUT_DIR = r"C:\Users\ErikWang\Documents\new_poli_analysis\Politician_Quarterly_Analysis\Input"
os.makedirs(INPUT_DIR, exist_ok=True)

def get_stock_prices_yf(ticker, transaction_date):
    """
    Get stock opening price on transaction date and closing price on March 28, 2025
    using Yahoo Finance
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        
        # Convert transaction_date to datetime if it's not already
        if isinstance(transaction_date, str):
            transaction_date = pd.to_datetime(transaction_date)
        
        # Get historical data covering our full date range
        hist = stock.history(start=transaction_date.strftime('%Y-%m-%d'), 
                           end='2025-03-28')
        
        if hist.empty:
            print(f"No data found for {ticker}")
            return None, None
            
        # Get entry price (opening price on transaction date)
        entry_date = transaction_date.strftime('%Y-%m-%d')
        
        # Look for the next available trading day if no data on exact transaction date
        if entry_date not in hist.index:
            # Get the first available date after the transaction
            next_dates = hist.index[hist.index >= entry_date]
            if len(next_dates) > 0:
                entry_date = next_dates[0].strftime('%Y-%m-%d')
                print(f"Using next trading day {entry_date} for {ticker} (transaction was on {transaction_date.strftime('%Y-%m-%d')})")
            else:
                print(f"No trading data available after {entry_date} for {ticker}")
                return None, None
        
        entry_price = hist.loc[entry_date, 'Open']
            
        # Get exit price (last available closing price)
        if len(hist) > 0:
            exit_price = hist.iloc[-1]['Close']
        else:
            print(f"No closing price data for {ticker}")
            return None, None
        
        # Verify we have valid prices
        if pd.isna(entry_price) or pd.isna(exit_price):
            print(f"Invalid prices for {ticker}: Entry={entry_price}, Exit={exit_price}")
            return None, None
            
        return float(entry_price), float(exit_price)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def save_progress(results, failed_tickers, last_processed_idx, filename='progress_data.json'):
    """Save current progress to a file"""
    # Convert DataFrame-style records to serializable format
    serializable_results = []
    for result in results:
        # Convert any Timestamp objects to strings
        result = result.copy()  # Make a copy to avoid modifying the original
        for key, value in result.items():
            if isinstance(value, pd.Timestamp):
                result[key] = value.strftime('%Y-%m-%d')
        serializable_results.append(result)
    
    progress_data = {
        'results': serializable_results,
        'failed_tickers': list(set(failed_tickers)),
        'last_processed_idx': int(last_processed_idx),  # Ensure it's a regular int
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results to CSV in the input directory
    if results:
        output_path = os.path.join(INPUT_DIR, 'partial_results.csv')
        pd.DataFrame(results).to_csv(output_path, index=False)
    
    # Save progress metadata
    output_path = os.path.join(INPUT_DIR, filename)
    with open(output_path, 'w') as f:
        json.dump(progress_data, f)
    
    print(f"\nProgress saved: {len(results)} processed trades, {len(set(failed_tickers))} failed tickers")
    print(f"Last processed index: {last_processed_idx}")
    print(f"Saved to: {INPUT_DIR}")

def load_progress(filename='progress_data.json'):
    """Load previously saved progress"""
    try:
        progress_file = os.path.join(INPUT_DIR, filename)
        results_file = os.path.join(INPUT_DIR, 'partial_results.csv')
        
        if os.path.exists(progress_file) and os.path.exists(results_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            # Load partial results from CSV to maintain data types
            results_df = pd.read_csv(results_file)
            # Convert date strings back to Timestamps
            if 'TransactionDate' in results_df.columns:
                results_df['TransactionDate'] = pd.to_datetime(results_df['TransactionDate'])
            results = results_df.to_dict('records')
            
            print(f"\nFound saved progress from {progress_data['timestamp']}")
            print(f"Loaded {len(results)} processed trades")
            print(f"Will resume from index {progress_data['last_processed_idx'] + 1}")
            
            return results, progress_data['failed_tickers'], progress_data['last_processed_idx']
    except Exception as e:
        print(f"Error loading progress: {str(e)}")
    
    return [], [], -1

def calculate_trade_performance(filtered_data, save_interval=50):
    """
    Calculate performance for each trade with progress saving
    """
    # Try to load previous progress
    results, failed_tickers, last_processed_idx = load_progress()
    
    total_trades = len(filtered_data)
    retry_tickers = []
    
    print(f"\nCalculating performance for {total_trades} trades...")
    
    # Skip already processed trades if resuming
    start_idx = last_processed_idx + 1
    
    for idx, trade in filtered_data.iloc[start_idx:].iterrows():
        if idx % 10 == 0:  # Progress update every 10 trades
            print(f"Processing trade {idx + 1} of {total_trades}")
        
        if idx % 100 == 0 and idx > 0:  # Add delay every 100 requests
            print("Rate limit pause - waiting 10 seconds...")
            time.sleep(10)
        
        if idx % save_interval == 0 and idx > start_idx:  # Save progress periodically
            save_progress(results, failed_tickers, idx)
            
        ticker = trade['Ticker']
        # Use TransactionDate instead of Date
        transaction_date = trade['TransactionDate']
        entry_price, exit_price = get_stock_prices_yf(ticker, transaction_date)
        
        if entry_price and exit_price:
            # Calculate percent change
            pct_change = ((exit_price - entry_price) / entry_price) * 100
            
            # Adjust for transaction type (multiply by -1 for sales/shorts)
            if trade['Transaction'].lower() in ['sale', 'short']:
                pct_change *= -1
                
            # Create result dictionary with all original columns
            result = trade.to_dict()
            # Add the performance metrics
            result.update({
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'PercentChange': pct_change
            })
            # Remove sensitive columns if they exist
            for col in ['ExcessReturn', 'PriceChange', 'SPYChange']:
                result.pop(col, None)
                
            results.append(result)
        else:
            if "Rate limited" in str(failed_tickers):
                retry_tickers.append((ticker, trade))
                print(f"Will retry {ticker} later due to rate limiting")
                time.sleep(5)
            else:
                failed_tickers.append(ticker)
    
    # Save final progress before retries
    save_progress(results, failed_tickers, total_trades - 1)
    
    # Retry rate-limited tickers
    if retry_tickers:
        print(f"\nRetrying {len(retry_tickers)} rate-limited tickers after 60 second pause...")
        time.sleep(60)
        
        for ticker, trade in retry_tickers:
            print(f"Retrying {ticker}...")
            entry_price, exit_price = get_stock_prices_yf(ticker, trade['TransactionDate'])
            
            if entry_price and exit_price:
                pct_change = ((exit_price - entry_price) / entry_price) * 100
                if trade['Transaction'].lower() in ['sale', 'short']:
                    pct_change *= -1
                    
                # Create result dictionary with all original columns
                result = trade.to_dict()
                # Add the performance metrics
                result.update({
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PercentChange': pct_change
                })
                # Remove sensitive columns if they exist
                for col in ['ExcessReturn', 'PriceChange', 'SPYChange']:
                    result.pop(col, None)
                    
                results.append(result)
            else:
                failed_tickers.append(ticker)
            
            time.sleep(5)
    
    # Save final results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(INPUT_DIR, 'final_results.csv')
    results_df.to_csv(output_path, index=False)
    
    # Print summary of failed tickers
    if failed_tickers:
        print(f"\nWarning: Could not get data for {len(set(failed_tickers))} unique tickers:")
        print(sorted(set(failed_tickers)))
    
    # Add summary statistics
    if not results_df.empty:
        print("\nPerformance Summary:")
        print(f"Successfully processed: {len(results_df)} of {total_trades} trades")
        print(f"Average Return: {results_df['PercentChange'].mean():.2f}%")
        print(f"Median Return: {results_df['PercentChange'].median():.2f}%")
        print("\nTop 5 Performing Trades:")
        print(results_df.nlargest(5, 'PercentChange')[
            ['Representative', 'Ticker', 'Transaction', 'PercentChange', 'EntryPrice', 'ExitPrice']
        ].to_string())
        print("\nBottom 5 Performing Trades:")
        print(results_df.nsmallest(5, 'PercentChange')[
            ['Representative', 'Ticker', 'Transaction', 'PercentChange', 'EntryPrice', 'ExitPrice']
        ].to_string())
    
    return results_df

def save_raw_data(df, output_dir):
    """Save the filtered data to CSV for reference, excluding only sensitive columns."""
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # List of columns to exclude
    exclude_columns = ['ExcessReturn', 'PriceChange', 'SPYChange']
    
    # Remove sensitive columns if they exist
    for col in exclude_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Excluded sensitive column: {col}")
    
    # Save to CSV with all remaining columns
    output_path = os.path.join(output_dir, 'party_performance_raw_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved filtered raw data to: {output_path}")
    
    # Print verification of the columns included
    print(f"Raw data CSV includes {len(df)} rows with {len(df.columns)} columns")
    print(f"Excluded columns: {', '.join(exclude_columns)}")
    
    return output_path

print(f"API Key loaded: {'Yes' if QUIVER_API_KEY else 'No'}")

try:
    # Use the bulk congress trading endpoint
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {QUIVER_API_KEY}'
    }
    
    url = 'https://api.quiverquant.com/beta/bulk/congresstrading'
    
    print("Fetching data from bulk congress trading endpoint...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # Convert JSON response to DataFrame
        raw_data = pd.DataFrame(response.json())
        
        # Convert TransactionDate to datetime
        raw_data['TransactionDate'] = pd.to_datetime(raw_data['TransactionDate'])
        
        # Define date ranges
        q4_2024_start = pd.Timestamp('2024-10-01')
        q4_2024_end = pd.Timestamp('2024-12-31')
        q1_2025_start = pd.Timestamp('2025-01-01')
        q1_2025_end = pd.Timestamp('2025-03-28')  # Today
        
        # Filter for Q4 2024 and Q1 2025
        mask = (
            ((raw_data['TransactionDate'] >= q4_2024_start) & (raw_data['TransactionDate'] <= q4_2024_end)) |
            ((raw_data['TransactionDate'] >= q1_2025_start) & (raw_data['TransactionDate'] <= q1_2025_end))
        )
        filtered_data = raw_data[mask].copy()
        
        # Add quarter column for easier analysis
        filtered_data['Quarter'] = 'Other'
        filtered_data.loc[(filtered_data['TransactionDate'] >= q4_2024_start) & 
                         (filtered_data['TransactionDate'] <= q4_2024_end), 'Quarter'] = '2024-Q4'
        filtered_data.loc[(filtered_data['TransactionDate'] >= q1_2025_start) & 
                         (filtered_data['TransactionDate'] <= q1_2025_end), 'Quarter'] = '2025-Q1'
        
        print("\n=== FILTERED DATA SUMMARY ===")
        print(f"Total records: {len(filtered_data)}")
        print(f"Q4 2024 records: {len(filtered_data[filtered_data['Quarter'] == '2024-Q4'])}")
        print(f"Q1 2025 records: {len(filtered_data[filtered_data['Quarter'] == '2025-Q1'])}")
        
        print("\nColumns:")
        print(filtered_data.columns.tolist())
        
        print("\nFirst 10 rows of filtered data:")
        print(filtered_data.head(10))
        
        print("\nDate range in filtered data:")
        print(f"Earliest date: {filtered_data['TransactionDate'].min()}")
        print(f"Latest date: {filtered_data['TransactionDate'].max()}")
        
        print("\nUnique Representatives by quarter:")
        for quarter in ['2024-Q4', '2025-Q1']:
            quarter_reps = filtered_data[filtered_data['Quarter'] == quarter]['Representative'].unique()
            print(f"\n{quarter} ({len(quarter_reps)} representatives):")
            print(sorted(quarter_reps))
        
        # Check specifically for Pelosi's trades
        pelosi_data = filtered_data[filtered_data['Representative'].str.contains('Pelosi', case=False, na=False)]
        print("\nPelosi's trades by quarter:")
        for quarter in ['2024-Q4', '2025-Q1']:
            quarter_trades = pelosi_data[pelosi_data['Quarter'] == quarter]
            print(f"\n{quarter} ({len(quarter_trades)} trades):")
            if not quarter_trades.empty:
                print(quarter_trades[['TransactionDate', 'Ticker', 'Transaction', 'Range']].sort_values('TransactionDate'))
        
        # Calculate trade performance using Yahoo Finance data
        print("\nCalculating trade performance using Yahoo Finance...")
        performance_data = calculate_trade_performance(filtered_data)
        
        # Save both raw and performance data
        filtered_data.to_csv(os.path.join(INPUT_DIR, 'congress_trading_2024Q4_2025Q1.csv'), index=False)
        performance_data.to_csv(os.path.join(INPUT_DIR, 'congress_trading_performance_2024Q4_2025Q1.csv'), index=False)
        
        # Show performance by representative
        if not performance_data.empty:
            print("\nPerformance by Representative:")
            rep_performance = performance_data.groupby('Representative').agg({
                'PercentChange': ['mean', 'count'],
                'Ticker': 'nunique'
            }).round(2)
            rep_performance.columns = ['Avg Return %', 'Num Trades', 'Num Stocks']
            rep_performance = rep_performance.sort_values('Avg Return %', ascending=False)
            print(rep_performance)
        
        # Save raw data
        save_raw_data(filtered_data, INPUT_DIR)
        
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {str(e)}") 