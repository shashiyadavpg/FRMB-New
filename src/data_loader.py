import yfinance as yf
import pandas as pd
import os
import time

class DataLoader:
    def __init__(self, data_dir='data/raw', delay=0.5):
        self.data_dir = data_dir
        self.delay = delay
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def fetch_company_data(self, ticker, start_date="2015-01-01", end_date="2024-12-31"):
        """
        Fetches historical price data and financials for a given ticker.
        """
        print(f"Fetching data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Market Data (History)
            history = stock.history(start=start_date, end=end_date)
            if history.empty:
                print(f"Warning: No price data found for {ticker}")
            
            # 2. Financials
            # yfinance returns DataFrame with columns as dates
            balance_sheet = stock.balance_sheet
            income_stmt = stock.financials
            cashflow = stock.cashflow
            
            # Save raw data
            save_path = os.path.join(self.data_dir, f"{ticker}_market.csv")
            history.to_csv(save_path)
            
            # Save Financials
            stock.balance_sheet.T.to_csv(os.path.join(self.data_dir, f"{ticker}_balance_sheet.csv"))
            stock.financials.T.to_csv(os.path.join(self.data_dir, f"{ticker}_income_stmt.csv"))
            stock.cashflow.T.to_csv(os.path.join(self.data_dir, f"{ticker}_cashflow.csv"))
            
            # 3. Key Statistics / Info
            # (Note: 'info' is a dict, fetching it can be slow, handle carefully)
            # info = stock.info 

            # Transpose financial statements to have dates as index
            return {
                'history': history,
                'balance_sheet': balance_sheet.T,
                'income_stmt': income_stmt.T,
                'cashflow': cashflow.T
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
        finally:
            time.sleep(self.delay)

    def batch_fetch(self, tickers_list):
        """
        Fetches data for a list of tickers.
        """
        results = {}
        for ticker in tickers_list:
            data = self.fetch_company_data(ticker)
            if data:
                results[ticker] = data
        return results

if __name__ == "__main__":
    # Test with a few tickers
    loader = DataLoader()
    test_tickers = ['AAPL', 'MSFT', 'F', 'AMC'] # Mixed bag
    loader.batch_fetch(test_tickers)
