import pandas as pd
import json
import os
import xlsxwriter

class ReportGenerator:
    def __init__(self, output_dir='outputs', model_file='model_output_scored.csv'):
        self.output_dir = output_dir
        self.model_file = os.path.join(output_dir, model_file)
        self.excel_path = os.path.join(output_dir, 'Corporate_PD_Model_Dashboard_v2.xlsx')
    
    def generate_excel(self):
        print("Generating Excel Dashboard...")
        if not os.path.exists(self.model_file):
            print("Model output file not found. Run training first.")
            return

        df = pd.read_csv(self.model_file)
        
        # Load Metrics & Coefficients
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        coef_path = os.path.join(self.output_dir, 'logistic_coefficients.csv')
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        coef_df = pd.DataFrame()
        if os.path.exists(coef_path):
            coef_df = pd.read_csv(coef_path)

        # Create Excel Writer
        writer = pd.ExcelWriter(self.excel_path, engine='xlsxwriter')
        workbook = writer.book

        # Formats
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#DCE6F1', 'border': 1})
        subhead_fmt = workbook.add_format({'bold': True, 'font_color': '#1F497D', 'bottom': 1})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        num_fmt = workbook.add_format({'num_format': '0.00'})
        high_risk_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        
        # --- SHEET 1: Methodology & Overview ---
        ws_intro = workbook.add_worksheet('Methodology & Overview')
        
        ws_intro.write(0, 0, "Global Corporate Probability-of-Default (PD) Model", workbook.add_format({'bold': True, 'font_size': 16}))
        ws_intro.write(2, 0, "1. Project Objective", subhead_fmt)
        ws_intro.write(3, 0, "To estimate the 1-year Probability of Default (PD) for public companies using financial ratios and market signals.")
        
        ws_intro.write(5, 0, "2. Modelling Workflow", subhead_fmt)
        steps = [
            "Step 1: Data Ingestion (Yahoo Finance) - Price history, Balance Sheet, Income Statement.",
            "Step 2: Feature Engineering - Calc Distess Scores (Altman Z), Volatility, Leverage Trends.",
            "Step 3: Training - Logistic Regression, Random Forest, XGBoost trained on historical data.",
            "Step 4: Scoring - Models predict PD for the most recent fiscal year."
        ]
        for i, step in enumerate(steps):
            ws_intro.write(6+i, 0, step)
            
        ws_intro.write(11, 0, "3. Model Performance (Test Set)", subhead_fmt)
        row = 12
        for model_name, scores in metrics.items():
            ws_intro.write(row, 0, model_name)
            ws_intro.write(row, 1, f"AUC: {scores.get('AUC', 0):.4f}")
            ws_intro.write(row, 2, f"KS: {scores.get('KS', 0):.4f}")
            row += 1

        # --- SHEET 2: Model Specs (Coefficients) ---
        if not coef_df.empty:
            coef_df.to_excel(writer, sheet_name='Model Specs', index=False, startrow=1)
            ws_specs = writer.sheets['Model Specs']
            ws_specs.write(0, 0, "Logistic Regression Coefficients (Impact on Log-Odds)", subhead_fmt)
            ws_specs.set_column(0, 0, 20)
            ws_specs.set_column(1, 1, 15)
            
            # Add explanation
            ws_specs.write(0, 3, "Interpretation:", subhead_fmt)
            ws_specs.write(1, 3, "Positive Coef (+):  Increases Default Risk")
            ws_specs.write(2, 3, "Negative Coef (-):  Decreases Default Risk (Protective)")

        # --- SHEET 3: Step-by-Step Calc Example ---
        ws_calc = workbook.add_worksheet('Calculation Example')
        ws_calc.write(0, 0, "Step-by-Step PD Calculation for a Single Company", workbook.add_format({'bold': True, 'font_size': 14}))
        
        # Pick a sample company (e.g., one with decent risk or just a random one)
        sample = df.iloc[0] # Default to first
        # Try to find a risky one for better demo
        risky_sample = df[df['PD_Logistic'] > 0.05]
        if not risky_sample.empty:
            sample = risky_sample.iloc[0]
            
        # Display Raw Inputs
        ws_calc.write(2, 0, "1. Raw Inputs (Financials & Market)", subhead_fmt)
        raw_cols = ['Ticker', 'Year', 'TotalAssets', 'TotalEquity', 'TotalDebt', 'EBIT', 'NetIncome', 'MarketCap']
        for i, col in enumerate(raw_cols):
            ws_calc.write(3, i, col, header_fmt)
            val = sample[col] if col in sample else 0
            ws_calc.write(4, i, val, num_fmt)
            
        # Display Computed Features
        ws_calc.write(6, 0, "2. Computed Risk Features (Standardized for Model)", subhead_fmt)
        feat_cols = ['DebtToEquity', 'ROA', 'Z_Score', 'Volatility_1Y']
        for i, col in enumerate(feat_cols):
            ws_calc.write(7, i, col, header_fmt)
            val = sample[col] if col in sample else 0
            ws_calc.write(8, i, val, num_fmt)
            
        # Show Logistic Formula
        ws_calc.write(10, 0, "3. Logistic Model Calculation", subhead_fmt)
        ws_calc.write(11, 0, "Log-Odds (L) = Intercept + Sum(Coef_i * Feature_i)")
        ws_calc.write(12, 0, "PD = 1 / (1 + exp(-L))")
        
        ws_calc.write(14, 0, "Model Output for this Company:", workbook.add_format({'bold': True}))
        ws_calc.write(14, 1, sample['PD_Logistic'], pct_fmt)
        ws_calc.write(14, 2, "<- This is the final Probability of Default")

        # --- SHEET 4: Dashboard ---
        # Write the main data
        cols = ['Ticker', 'Company', 'Sector', 'Region', 'Year', 'PD_XGB', 'PD_Logistic', 'Manual_Rating', 'Z_Score', 'DebtToEquity', 'ROA']
        display_df = df[cols].copy()
        display_df.rename(columns={'PD_XGB': 'PD (XGBoost)', 'PD_Logistic': 'PD (Logistic)'}, inplace=True)
        
        display_df.to_excel(writer, sheet_name='Dashboard', index=False)
        ws_dash = writer.sheets['Dashboard']
        
        # Apply Conditional Formatting
        pd_col_idx = 5
        last_row = len(display_df) + 1
        ws_dash.conditional_format(1, pd_col_idx, last_row, pd_col_idx, {
            'type': '3_color_scale',
            'min_color': '#63BE7B', 'mid_color': '#FFEB84', 'max_color': '#F8696B'
        })
        
        ws_dash.set_column(0, 0, 10) # Ticker
        ws_dash.set_column(1, 1, 25) # Company
        ws_dash.set_column(5, 6, 12, pct_fmt) # PDs
        ws_dash.autofilter(0, 0, last_row, len(cols)-1)
        
        # --- SHEET 5: Top Risky ---
        max_year = df['Year'].max()
        latest_df = df[df['Year'] == max_year].copy()
        top_risky = latest_df.sort_values(by='PD_XGB', ascending=False).head(20)
        top_risky[['Ticker', 'Company', 'PD_XGB', 'Z_Score', 'DebtToEquity']].to_excel(writer, sheet_name='Top Risky', index=False)
        
        writer.close()
        print(f"Workbook saved to {self.excel_path}")

if __name__ == "__main__":
    gen = ReportGenerator()
    gen.generate_excel()
