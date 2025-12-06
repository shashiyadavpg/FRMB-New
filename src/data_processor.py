import pandas as pd
import numpy as np
import os
import glob

class DataProcessor:
    def __init__(self, raw_dir='data/raw', metadata_path='data/companies.csv'):
        self.raw_dir = raw_dir
        self.metadata_path = metadata_path
        self.companies_df = pd.read_csv(metadata_path)

    def load_and_consolidate(self):
        """
        Reads raw CSVs for each ticker and merges them into a single panel dataset.
        Aggregates daily market data to annual frequency.
        """
        consolidated_records = []

        for _, row in self.companies_df.iterrows():
            ticker = row['Ticker']
            company_info = row.to_dict()
            
            print(f"Processing {ticker}...")
            
            # Paths
            market_path = os.path.join(self.raw_dir, f"{ticker}_market.csv")
            bs_path = os.path.join(self.raw_dir, f"{ticker}_balance_sheet.csv")
            is_path = os.path.join(self.raw_dir, f"{ticker}_income_stmt.csv")
            cf_path = os.path.join(self.raw_dir, f"{ticker}_cashflow.csv")

            # Load Data
            try:
                if not os.path.exists(market_path):
                    print(f"  Missing market data for {ticker}")
                    continue

                market_df = pd.read_csv(market_path)
                market_df['Date'] = pd.to_datetime(market_df['Date'], utc=True).dt.tz_localize(None)
                market_df.set_index('Date', inplace=True)

                # Load Financials (Annual)
                # Financials might not exist if yfinance failed
                financials_combined = pd.DataFrame()
                
                if os.path.exists(bs_path) and os.path.exists(is_path):
                    bs = pd.read_csv(bs_path, index_col=0, parse_dates=True)
                    inc = pd.read_csv(is_path, index_col=0, parse_dates=True)
                    cf = pd.read_csv(cf_path, index_col=0, parse_dates=True) if os.path.exists(cf_path) else pd.DataFrame()
                    
                    # Convert index to datetime just in case
                    bs.index = pd.to_datetime(bs.index)
                    inc.index = pd.to_datetime(inc.index)
                    if not cf.empty: cf.index = pd.to_datetime(cf.index)

                    # Merge Financials by Index (Date)
                    # yfinance dates are usually year-end. We merge on index.
                    # Use 'outer' to keep all years
                    financials_combined = bs.join(inc, lsuffix='_bs', rsuffix='_is', how='outer')
                    if not cf.empty:
                        financials_combined = financials_combined.join(cf, rsuffix='_cf', how='outer')
                
                else:
                    print(f"  Missing financials for {ticker}, skipping...")
                    continue

                # Now merge Market Features into Financial Years
                # We need to compute annual market stats for each financial year ending
                
                for date in financials_combined.index:
                    # Define the "Fiscal Year" period as (Date - 365 days) to Date
                    # Or simpler: Just take the market metrics at that Date (Price) and Volatility over prev year
                    
                    end_date = date
                    start_date = date - pd.Timedelta(days=365)
                    
                    mask = (market_df.index > start_date) & (market_df.index <= end_date)
                    period_market = market_df.loc[mask]
                    
                    if period_market.empty:
                        continue

                    # Market features
                    price_start = period_market['Close'].iloc[0]
                    price_end = period_market['Close'].iloc[-1]
                    price_return = (price_end / price_start) - 1
                    volatility = period_market['Close'].pct_change().std() * np.sqrt(252)
                    avg_volume = period_market['Volume'].mean()
                    
                    # Create Record
                    record = company_info.copy()
                    record['ReportDate'] = date
                    record['Year'] = date.year
                    
                    # Add Financials (Flattening)
                    # This maps common yfinance names to our schema
                    # Note: Yfinance column names vary wildly. We try standard ones.
                    
                    # Map common fields safely
                    def get_val(df, col_variations):
                        for c in col_variations:
                            if c in df.columns:
                                val = df.loc[date, c]
                                return val if not pd.isna(val) else 0.0
                        return 0.0

                    record['TotalAssets'] = get_val(financials_combined, ['Total Assets', 'Assets'])
                    record['TotalLiabilities'] = get_val(financials_combined, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
                    record['TotalEquity'] = get_val(financials_combined, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])
                    record['TotalDebt'] = get_val(financials_combined, ['Total Debt'])
                    record['Cash'] = get_val(financials_combined, ['Cash And Cash Equivalents', 'Cash Financial'])
                    record['Revenue'] = get_val(financials_combined, ['Total Revenue', 'Operating Revenue'])
                    record['EBITDA'] = get_val(financials_combined, ['EBITDA', 'Normalized EBITDA'])
                    record['EBIT'] = get_val(financials_combined, ['EBIT'])
                    record['NetIncome'] = get_val(financials_combined, ['Net Income', 'Net Income Common Stockholders'])
                    record['InterestExpense'] = get_val(financials_combined, ['Interest Expense'])
                    record['OperatingCashFlow'] = get_val(financials_combined, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
                    
                    # Market Features
                    record['MarketCap'] = price_end * get_val(financials_combined, ['Share Issued']) # Approx
                    record['SharePrice'] = price_end
                    record['Return_1Y'] = price_return
                    record['Volatility_1Y'] = volatility
                     
                    consolidated_records.append(record)
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # Create DataFrame
        final_df = pd.DataFrame(consolidated_records)
        
        # Save
        os.makedirs('data/processed', exist_ok=True)
        save_path = 'data/processed/consolidated_data.csv'
        final_df.to_csv(save_path, index=False)
        print(f"Consolidated data saved to {save_path} with {len(final_df)} records.")
        return final_df

if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_and_consolidate()
