import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, input_path='data/processed/consolidated_data.csv', output_path='data/processed/model_input.csv'):
        self.input_path = input_path
        self.output_path = output_path
        
        # Hardcoded default years for known defaulters in our list
        self.default_years = {
            'BBBY': 2023,
            'SVB': 2023,
            'SBNY': 2023,
            'FRC': 2023,
            'RAD': 2023, # Bankruptcy filing
            'AMC': 2025, # High distress placeholder
            'RCOM.NS': 2019,
            'DHFL.NS': 2019,
            'JPPOWER.NS': 2016, # Restructuring
            'IDEA.NS': 2024, # High distress
            'RPOWER.NS': 2020,
        }

    def compute_features(self):
        print("Loading consolidated data...")
        df = pd.read_csv(self.input_path)
        
        # 1. Financial Health Ratios
        print("Computing financial ratios...")
        # Handle division by zero
        EPS = 1e-6
        
        df['ROA'] = df['NetIncome'] / (df['TotalAssets'] + EPS)
        df['ROE'] = df['NetIncome'] / (df['TotalEquity'] + EPS)
        df['NetMargin'] = df['NetIncome'] / (df['Revenue'] + EPS)
        df['EBITDAMargin'] = df['EBITDA'] / (df['Revenue'] + EPS)
        df['DebtToEquity'] = df['TotalDebt'] / (df['TotalEquity'] + EPS)
        df['DebtToAssets'] = df['TotalDebt'] / (df['TotalAssets'] + EPS)
        df['InterestCoverage'] = df['EBIT'] / (df['InterestExpense'] + EPS)
        df['OCF_to_Debt'] = df['OperatingCashFlow'] / (df['TotalDebt'] + EPS)
        df['CurrentRatio'] = df['TotalAssets'] / (df['TotalLiabilities'] + EPS) # Approximation since we lacked Current Assets
        
        # 2. Market Signals
        df['MarketCap_to_Assets'] = df['MarketCap'] / (df['TotalAssets'] + EPS)
        # Return_1Y and Volatility_1Y are already in raw data
        
        # 3. Distress Models
        # Altman Z-Score (Public Service Variance)
        # Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
        # X1 = Working Capital / Total Assets (Using (Assets-Liab)/Assets approx)
        # X2 = Retained Earnings / Total Assets (Proxy: Net Income/Assets * Year, weak proxy, use ROA) -> actually RE is cumulative. We use Equity/Assets as proxy for X4.
        
        # Simplified Z-Score for non-manufacturers (Z''):
        # Z'' = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4
        # X1 = (Current Assets - Current Liabilities) / Total Assets. ~ (Equity? No) -> Use WorkingCapital proxy
        # X2 = Retained Earnings / Total Assets. ~ Use 0.5 * ROA ?
        # X3 = EBIT / Total Assets.
        # X4 = Book Value of Equity / Total Liabilities.
        
        # Let's use a standard custom proxy based on available fields:
        # Z_Proxy = 1.2*(WorkingCap/Assets) + 3.3*(EBIT/Assets) + 0.6*(MarketCap/Liabilities) + 1.0*(Sales/Assets)
        # WorkingCapital Proxy = (Assets - Liabilities) (This is Equity, but assuming LongTerm debt exists, it's different. We lack Current Assets).
        # We will use Equity/Assets as X1 proxy.
        
        df['X1'] = df['TotalEquity'] / (df['TotalAssets'] + EPS)
        df['X2'] = df['EBIT'] / (df['TotalAssets'] + EPS)
        df['X3'] = df['MarketCap'] / (df['TotalLiabilities'] + EPS)
        df['X4'] = df['Revenue'] / (df['TotalAssets'] + EPS)
        
        df['Z_Score'] = 1.2*df['X1'] + 3.3*df['X2'] + 0.6*df['X3'] + 1.0*df['X4']
        
        # Ohlson O-Score Implementation (Simplified)
        # O = -1.32 - .407*log(TotalAssets/GNP) + 6.03*(TL/TA) - 1.43*(WC/TA) + .076*CL/CA - 1.72*OENEG - 2.37*NINEG - 1.83*FUTL + .285*INTWO - .521*CHIN
        # We will use a reduced version:
        # O_Proxy = (TotalLiabilities/TotalAssets) * 5.0 - (NetIncome/TotalAssets) * 3.0
        df['O_Score'] = (df['TotalLiabilities']/(df['TotalAssets']+EPS)) * 5.0 - df['ROA'] * 3.0

        # 4. Trends & Lags
        # Need to sort by Company and Year
        df = df.sort_values(by=['Ticker', 'Year'])
        
        df['ROA_Lag1'] = df.groupby('Ticker')['ROA'].shift(1)
        df['DE_Lag1'] = df.groupby('Ticker')['DebtToEquity'].shift(1)
        df['Slope_Leverage'] = df['DebtToAssets'] - df.groupby('Ticker')['DebtToAssets'].shift(1)
        
        # Fill NA lags with current (first year)
        df[['ROA_Lag1', 'DE_Lag1', 'Slope_Leverage']] = df[['ROA_Lag1', 'DE_Lag1', 'Slope_Leverage']].fillna(0)

        # 5. Label Generation
        print("Generating labels...")
        def get_label(row):
            # If the company is in our default list
            if row['Ticker'] in self.default_years:
                default_year = self.default_years[row['Ticker']]
                # If current year is the year JUST BEFORE default or THE year of default
                if row['Year'] == default_year or row['Year'] == default_year - 1:
                    return 1
                # If current year is after default, it's post-default (drop or keep as 1? Usually drop)
                if row['Year'] > default_year:
                    return np.nan # Exclude post-default years
            
            # If Rating is D or C in the current year (Proxy)
            # We assume the 'Manual_Rating' is static. If it's D, all years might be bad? No.
            # We trust the manual default_years list for the actual Event.
            return 0

        df['default_within_1y'] = df.apply(get_label, axis=1)
        
        # Drop rows where label is NaN (post-default)
        df = df.dropna(subset=['default_within_1y'])
        
        # Final cleanup
        # Drop rows with 0 Assets or Revenue (bad data)
        df = df[df['TotalAssets'] > 0]
        
        print(f"Features computed. Saving {len(df)} rows to {self.output_path}")
        df.to_csv(self.output_path, index=False)

if __name__ == "__main__":
    eng = FeatureEngineer()
    eng.compute_features()
