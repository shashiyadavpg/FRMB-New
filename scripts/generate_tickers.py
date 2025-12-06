import pandas as pd
import os

def generate_company_list():
    # Define a diverse list of companies (US & India) to ensure >100 entries
    # Structure: Ticker, Company, Sector, Region, Estimated_Rating, Default_Event (0/1)
    
    data = [
        # --- US TECH (Generally High Rating) ---
        ('AAPL', 'Apple Inc.', 'Technology', 'US', 'AA+', 0),
        ('MSFT', 'Microsoft', 'Technology', 'US', 'AAA', 0),
        ('GOOGL', 'Alphabet', 'Technology', 'US', 'AA+', 0),
        ('AMZN', 'Amazon', 'Technology', 'US', 'AA', 0),
        ('NVDA', 'Nvidia', 'Technology', 'US', 'A', 0),
        ('META', 'Meta Platforms', 'Technology', 'US', 'A+', 0),
        ('TSLA', 'Tesla', 'Auto', 'US', 'BBB', 0),
        ('ORCL', 'Oracle', 'Technology', 'US', 'BBB+', 0),
        ('CSCO', 'Cisco', 'Technology', 'US', 'AA-', 0),
        ('INTC', 'Intel', 'Technology', 'US', 'A-', 0),
        ('IBM', 'IBM', 'Technology', 'US', 'A-', 0),
        ('ADBE', 'Adobe', 'Technology', 'US', 'A+', 0),
        ('CRM', 'Salesforce', 'Technology', 'US', 'A+', 0),
        ('AMD', 'AMD', 'Technology', 'US', 'A-', 0),
        ('QCOM', 'Qualcomm', 'Technology', 'US', 'A', 0),

        # --- US CONSUMER / RETAIL (Mixed) ---
        ('WMT', 'Walmart', 'Retail', 'US', 'AA', 0),
        ('TGT', 'Target', 'Retail', 'US', 'A', 0),
        ('COST', 'Costco', 'Retail', 'US', 'A+', 0),
        ('M', 'Macy\'s', 'Retail', 'US', 'BB', 0),
        ('KSS', 'Kohl\'s', 'Retail', 'US', 'BB-', 0),
        ('GPS', 'Gap', 'Retail', 'US', 'BB', 0),
        ('NKE', 'Nike', 'Consumer', 'US', 'AA-', 0),
        ('KO', 'Coca-Cola', 'Consumer', 'US', 'A+', 0),
        ('PEP', 'PepsiCo', 'Consumer', 'US', 'A+', 0),
        ('PG', 'P&G', 'Consumer', 'US', 'AA-', 0),
        
        # --- US ENERGY / INDUSTRIAL (Cyclical) ---
        ('XOM', 'Exxon Mobil', 'Energy', 'US', 'AA-', 0),
        ('CVX', 'Chevron', 'Energy', 'US', 'AA-', 0),
        ('BA', 'Boeing', 'Industrial', 'US', 'BBB-', 0), # Downgraded recently
        ('GE', 'General Electric', 'Industrial', 'US', 'BBB', 0),
        ('F', 'Ford', 'Auto', 'US', 'BB+', 0),
        ('GM', 'General Motors', 'Auto', 'US', 'BBB', 0),
        
        # --- DISTRESSED / RISKY / DEFAULTED (US) ---
        ('AMC', 'AMC Entertainment', 'Consumer', 'US', 'CCC', 1), # High distress
        ('GME', 'GameStop', 'Retail', 'US', 'B-', 0),
        ('CVNA', 'Carvana', 'Auto', 'US', 'CCC', 0),
        ('PTON', 'Peloton', 'Consumer', 'US', 'CCC+', 0),
        ('BBBY', 'Bed Bath & Beyond', 'Retail', 'US', 'D', 1), # Defaulted/Delisted (Data might be sparse)
        ('SVB', 'Silicon Valley Bank', 'Financial', 'US', 'D', 1), # Failed (Data until failure)
        ('FRC', 'First Republic', 'Financial', 'US', 'D', 1), # Failed
        ('SBNY', 'Signature Bank', 'Financial', 'US', 'D', 1), # Failed
        ('RAD', 'Rite Aid', 'Retail', 'US', 'D', 1),
        ('WEN', 'Wendy\'s', 'Consumer', 'US', 'B+', 0),

         # --- INDIA LARGE CAP (Stable) ---
        ('RELIANCE.NS', 'Reliance Industries', 'Energy', 'IN', 'AAA', 0),
        ('TCS.NS', 'TCS', 'Technology', 'IN', 'AAA', 0),
        ('HDFCBANK.NS', 'HDFC Bank', 'Financial', 'IN', 'AAA', 0),
        ('INFY.NS', 'Infosys', 'Technology', 'IN', 'AAA', 0),
        ('ICICIBANK.NS', 'ICICI Bank', 'Financial', 'IN', 'AAA', 0),
        ('HINDUNILVR.NS', 'HUL', 'Consumer', 'IN', 'AAA', 0),
        ('ITC.NS', 'ITC', 'Consumer', 'IN', 'AAA', 0),
        ('SBIN.NS', 'SBI', 'Financial', 'IN', 'AAA', 0),
        ('BHARTIARTL.NS', 'Bharti Airtel', 'Telecom', 'IN', 'AA+', 0),
        ('LICI.NS', 'LIC', 'Financial', 'IN', 'AAA', 0),
        ('LT.NS', 'L&T', 'Construction', 'IN', 'AAA', 0),
        ('TATAMOTORS.NS', 'Tata Motors', 'Auto', 'IN', 'AA', 0),
        ('SUNPHARMA.NS', 'Sun Pharma', 'Healthcare', 'IN', 'AA+', 0),
        ('MARUTI.NS', 'Maruti Suzuki', 'Auto', 'IN', 'AAA', 0),
        ('ULTRACEMCO.NS', 'UltraTech Cement', 'Materials', 'IN', 'AAA', 0),

        # --- INDIA MID/SMALL CAP (Mixed Ratings) ---
        ('VEDL.NS', 'Vedanta', 'Materials', 'IN', 'AA-', 0), 
        ('PNB.NS', 'PNB', 'Financial', 'IN', 'AA+', 0),
        ('IDFCFIRSTB.NS', 'IDFC First Bank', 'Financial', 'IN', 'AA', 0),
        ('YESBANK.NS', 'Yes Bank', 'Financial', 'IN', 'BBB', 0), # Was distressed
        ('IDEA.NS', 'Vodafone Idea', 'Telecom', 'IN', 'B-', 1), # High distress
        ('SUZLON.NS', 'Suzlon Energy', 'Energy', 'IN', 'B+', 0), # Recovering
        ('JPPOWER.NS', 'Jaiprakash Power', 'Energy', 'IN', 'B', 1), # Distressed history
        ('RPOWER.NS', 'Reliance Power', 'Energy', 'IN', 'C', 1),
        ('RCOM.NS', 'Reliance Comm', 'Telecom', 'IN', 'D', 1), # Defaulted
        ('DHFL.NS', 'DHFL', 'Financial', 'IN', 'D', 1), # Defaulted
        ('IBULHSGFIN.NS', 'Indiabulls Housing', 'Financial', 'IN', 'A+', 0), # Downgraded history
        
        # --- ADDITIONAL GLOBAL / MIX ---
        ('JPM', 'JPMorgan', 'Financial', 'US', 'A-', 0),
        ('V', 'Visa', 'Financial', 'US', 'AA-', 0),
        ('MA', 'Mastercard', 'Financial', 'US', 'AA-', 0),
        ('DIS', 'Disney', 'Media', 'US', 'A-', 0),
        ('NFLX', 'Netflix', 'Media', 'US', 'BBB+', 0),
        ('NIO', 'NIO', 'Auto', 'CN', 'B+', 0),
        ('BABA', 'Alibaba', 'Technology', 'CN', 'A+', 0),
        ('BIDU', 'Baidu', 'Technology', 'CN', 'A', 0),
        ('SONY', 'Sony', 'Consumer', 'JP', 'A', 0),
        ('TM', 'Toyota', 'Auto', 'JP', 'A+', 0),
        ('HMC', 'Honda', 'Auto', 'JP', 'A-', 0),
        ('SHEL', 'Shell', 'Energy', 'UK', 'AA-', 0),
        ('BP', 'BP', 'Energy', 'UK', 'A-', 0),
        ('DEO', 'Diageo', 'Consumer', 'UK', 'A', 0),
        ('UL', 'Unilever', 'Consumer', 'UK', 'A+', 0),
        ('NVS', 'Novartis', 'Healthcare', 'CH', 'AA', 0),
        ('AZN', 'AstraZeneca', 'Healthcare', 'UK', 'A', 0),
        ('SAP', 'SAP', 'Technology', 'DE', 'A', 0),
        ('SIEGY', 'Siemens', 'Industrial', 'DE', 'A+', 0),
        ('BUD', 'Anheuser-Busch', 'Consumer', 'BE', 'BBB+', 0),
        
        # --- MORE RISKY / B-Rated ---
        ('AAL', 'American Airlines', 'Travel', 'US', 'B-', 0),
        ('UAL', 'United Airlines', 'Travel', 'US', 'B+', 0),
        ('CCL', 'Carnival Corp', 'Travel', 'US', 'B', 0),
        ('RCL', 'Royal Caribbean', 'Travel', 'US', 'B+', 0),
        ('NCLH', 'Norwegian Cruise', 'Travel', 'US', 'B-', 0),
        ('MSTR', 'MicroStrategy', 'Tech', 'US', 'B-', 0), # High Volatility
        ('COIN', 'Coinbase', 'Financial', 'US', 'BB-', 0),
        
        # --- FILLERS TO REACH 100+ ---
        ('ABBV', 'AbbVie', 'Healthcare', 'US', 'A-', 0),
        ('PFE', 'Pfizer', 'Healthcare', 'US', 'A+', 0),
        ('JNJ', 'Johnson & Johnson', 'Healthcare', 'US', 'AAA', 0),
        ('LLY', 'Eli Lilly', 'Healthcare', 'US', 'A+', 0),
        ('MRK', 'Merck', 'Healthcare', 'US', 'A', 0),
        ('BMY', 'Bristol-Myers', 'Healthcare', 'US', 'A+', 0),
        ('AMGN', 'Amgen', 'Healthcare', 'US', 'A-', 0),
        ('GILD', 'Gilead', 'Healthcare', 'US', 'A-', 0),
        ('T', 'AT&T', 'Telecom', 'US', 'BBB', 0),
        ('VZ', 'Verizon', 'Telecom', 'US', 'BBB+', 0),
        ('TMUS', 'T-Mobile', 'Telecom', 'US', 'BBB-', 0),
        ('CMCSA', 'Comcast', 'Media', 'US', 'A-', 0),
        ('CHTR', 'Charter', 'Media', 'US', 'BB+', 0),
        
        # --- MORE INDIAN ---
        ('ADANIENT.NS', 'Adani Enterprises', 'Diverse', 'IN', 'A', 0),
        ('ADANIGREEN.NS', 'Adani Green', 'Energy', 'IN', 'A-', 0),
        ('ADANIPORTS.NS', 'Adani Ports', 'Infrastructure', 'IN', 'AA', 0),
        ('WIPRO.NS', 'Wipro', 'Technology', 'IN', 'A-', 0),
        ('TITAN.NS', 'Titan', 'Consumer', 'IN', 'AA+', 0),
        ('ASIANPAINT.NS', 'Asian Paints', 'Consumer', 'IN', 'AAA', 0),
        ('AXISBANK.NS', 'Axis Bank', 'Financial', 'IN', 'AA+', 0),
        ('BAJFINANCE.NS', 'Bajaj Finance', 'Financial', 'IN', 'AAA', 0),
        ('DMART.NS', 'Avenue Supermarts', 'Retail', 'IN', 'AA+', 0),
        ('ZOMATO.NS', 'Zomato', 'Tech', 'IN', 'AA-', 0)
    ]
    
    df = pd.DataFrame(data, columns=['Ticker', 'Company', 'Sector', 'Region', 'Manual_Rating', 'Default_Event'])
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/companies.csv', index=False)
    print(f"Generated data/companies.csv with {len(df)} companies.")

if __name__ == "__main__":
    generate_company_list()
