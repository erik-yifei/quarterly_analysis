# Politician Quarterly Analysis

This project analyzes congressional trading activity, comparing two time periods:
- **2024 Q4**: October 1, 2024 - December 31, 2024
- **2025 Q1**: January 1, 2025 - March 21, 2025

## Project Structure

Each analysis has its own folder with dedicated Python script, data, and visualizations:

- **Party_Performance/**: Analysis of portfolio performance by political party
  - `party_performance.py`: Script to analyze and visualize party performance
  - `party_performance_data.csv`: Performance metrics by party (generated)
  - `party_performance_raw_data.csv`: Filtered trading data (generated)
  - `party_performance_comparison.png`: Visualization of party performance (generated)

## Available Analyses

1. **Party Performance**: Compares the weighted portfolio performance of Democrats vs Republicans vs the market (SPY)
   - SPY reference prices:
     - 10/1/2024: $568.62
     - 12/31/2024: $586.08
     - 1/2/2025: $584.64
     - 3/21/2025: $563.98

## Setup

1. Create a `.env` file in the project root with your Quiver Quantitative API key:
   ```
   API_KEY=your_api_key_here
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run an analysis:
   ```bash
   python Party_Performance/party_performance.py
   ```

## Data Source

Congressional trading data is sourced from [Quiver Quantitative](https://www.quiverquant.com/).

## Security Note

- Never commit your API keys to version control
- The `.env` file is already included in `.gitignore` to prevent accidental commits 