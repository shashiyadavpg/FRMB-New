import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from data_loader import DataLoader

def main():
    if not os.path.exists('data/companies.csv'):
        print("Error: data/companies.csv not found.")
        return

    df = pd.read_csv('data/companies.csv')
    tickers = df['Ticker'].tolist()
    
    print(f"Starting data ingestion for {len(tickers)} companies...")
    
    loader = DataLoader(data_dir='data/raw')
    loader.batch_fetch(tickers)
    
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()
