import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, log_loss

class ForensicAuditor:
    def __init__(self, data_path='data/processed/model_input.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.report = []
        self.scores = {
            'Data_Quality': 100,
            'Model_Integrity': 100,
            'Statistical_Validity': 100,
            'Documentation': 100
        }

    def log(self, section, status, message, risk='Low'):
        entry = f"[{section}] [{status.upper()}] {message} (Risk: {risk})"
        print(entry)
        self.report.append(entry)
        if status == 'Fail':
            # Penalize
            if section == 'Data': self.scores['Data_Quality'] -= 10
            if section == 'Model': self.scores['Model_Integrity'] -= 15
            if section == 'Stats': self.scores['Statistical_Validity'] -= 10

    def audit_data_integrity(self):
        print("\n--- Auditing Data Integrity ---")
        if not os.path.exists(self.data_path):
            self.log('Data', 'Fail', 'Model input file missing!', 'Critical')
            return

        df = pd.read_csv(self.data_path)
        
        # 1. Company Count
        n_companies = df['Ticker'].nunique()
        if n_companies >= 100:
            self.log('Data', 'Pass', f"Company count verified: {n_companies} (>=100)")
        else:
            self.log('Data', 'Fail', f"Insufficient companies: {n_companies}", 'High')

        # 2. Years per Company
        yr_counts = df.groupby('Ticker')['Year'].count()
        avg_yrs = yr_counts.mean()
        min_yrs = yr_counts.min()
        if avg_yrs >= 5:
            self.log('Data', 'Pass', f"Average years per company: {avg_yrs:.1f}")
        else:
            self.log('Data', 'Fail', f"Low data depth. Avg years: {avg_yrs:.1f}", 'Medium')
            
        if min_yrs < 3:
             self.log('Data', 'Warning', f"Some companies have very few observations (Min: {min_yrs})", 'Low')

        # 3. Missing Values (Critical)
        critical_cols = ['TotalAssets', 'TotalDebt', 'NetIncome', 'MarketCap', 'default_within_1y']
        for col in critical_cols:
            if col in df.columns:
                n_missing = df[col].isnull().sum()
                if n_missing > 0:
                    self.log('Data', 'Fail', f"Missing values in {col}: {n_missing}", 'High')
                else:
                    self.log('Data', 'Pass', f"No missing values in {col}")
            else:
                 self.log('Data', 'Fail', f"Critical column {col} missing from schema", 'High')

        # 4. Outliers / Impossible Values
        if (df['TotalAssets'] < 0).any():
             self.log('Data', 'Fail', "Negative TotalAssets found", 'Critical')
        else:
             self.log('Data', 'Pass', "No negative Assets")

        # 5. Duplicates
        dupes = df.duplicated(subset=['Ticker', 'Year']).sum()
        if dupes > 0:
            self.log('Data', 'Fail', f"Duplicate Company-Year keys found: {dupes}", 'High')
        else:
            self.log('Data', 'Pass', "No duplicate keys")

        return df

    def audit_feature_engineering(self, df):
        print("\n--- Auditing Feature Engineering ---")
        # Recalculate D/E and ROA
        EPS = 1e-6
        # Check DebtToEquity
        recalc_de = df['TotalDebt'] / (df['TotalEquity'] + EPS)
        diff_de = (df['DebtToEquity'] - recalc_de).abs().max()
        if diff_de < 1e-4:
            self.log('Data', 'Pass', "DebtToEquity formula verification passed")
        else:
            self.log('Data', 'Fail', f"DebtToEquity mismatch. Max diff: {diff_de}", 'Critical')
            
        # Check Z-Score Logic
        # Z_Score = 1.2*X1 + 3.3*X2 + 0.6*X3 + 1.0*X4
        # Our X1 = Equity/Assets (In code), Standard Z X1 is WorkingCap/Assets
        # I need to verify if the code implementation matches the documented "Z_Score" column
        # Reconstruct exactly as per src/features.py logic
        
        x1 = df['TotalEquity'] / (df['TotalAssets'] + EPS)
        x2 = df['EBIT'] / (df['TotalAssets'] + EPS)
        x3 = df['MarketCap'] / (df['TotalLiabilities'] + EPS)
        x4 = df['Revenue'] / (df['TotalAssets'] + EPS)
        recalc_z = 1.2*x1 + 3.3*x2 + 0.6*x3 + 1.0*x4
        
        diff_z = (df['Z_Score'] - recalc_z).abs().max()
        if diff_z < 1e-4:
            self.log('Data', 'Pass', "Z_Score calculation matches source code logic")
            # Note: Whether source code logic is 'Correct' standard Z is another check.
            # Standard Z uses Working Capital. We used Equity. This is a "Methodology Deviation" but "Calculation Correctness" is Pass.
            self.log('Data', 'Info', "Z_Score uses Equity proxy for Working Capital (Methodology Deviation)", 'Low')
        else:
            self.log('Data', 'Fail', f"Z_Score failed recalculation. Max diff: {diff_z}", 'Major')

    def audit_model_integrity(self, df):
        print("\n--- Auditing Model Integrity ---")
        # Load scored output
        scored_path = os.path.join(self.output_dir, 'model_output_scored.csv')
        if not os.path.exists(scored_path):
            self.log('Model', 'Fail', 'Scored model output missing', 'Critical')
            return

        scored_df = pd.read_csv(scored_path)
        
        # Leakeage Check
        # Check if Training Data (Year < 2022) has 100% perfect predictions?
        # Overfitting check
        train_scored = scored_df[scored_df['Year'] < 2022]
        test_scored = scored_df[scored_df['Year'] >= 2022]
        
        train_auc = roc_auc_score(train_scored['default_within_1y'], train_scored['PD_Logistic'])
        test_auc = roc_auc_score(test_scored['default_within_1y'], test_scored['PD_Logistic'])
        
        self.log('Model', 'Info', f"Train AUC (Logistic): {train_auc:.4f}")
        self.log('Model', 'Info', f"Test AUC (Logistic): {test_auc:.4f}")
        
        if train_auc > 0.999:
             self.log('Model', 'Warning', "Train AUC is 1.0 or very close. Potential overfitting or data leakage.", 'Medium')
        
        if test_auc < 0.5:
             self.log('Model', 'Fail', "Test AUC < 0.5. Model is worse than random.", 'High')

        # Check calibration range
        max_pd = scored_df['PD_XGB'].max()
        min_pd = scored_df['PD_XGB'].min()
        self.log('Model', 'Info', f"PD Range: {min_pd:.4f} - {max_pd:.4f}")
        
        if max_pd < 0.2:
            self.log('Model', 'Warning', "Max PD is very low (< 20%). Model might be under-calibrated for distressed firms.", 'Medium')

    def run_full_audit(self):
        df = self.audit_data_integrity()
        if df is not None:
            self.audit_feature_engineering(df)
            self.audit_model_integrity(df)
            
        # Calculate Final Grade
        avg_score = sum(self.scores.values()) / 4
        if avg_score >= 90: grade = 'A'
        elif avg_score >= 80: grade = 'B'
        elif avg_score >= 70: grade = 'C'
        else: grade = 'D'
        
        print("\n--- FORENSIC AUDIT SUMMARY ---")
        print(f"Overall Grade: {grade} ({avg_score:.1f})")
        print(f"Scores: {self.scores}")
        
        # Save Report
        with open(os.path.join(self.output_dir, 'Forensic_Audit_Log.txt'), 'w') as f:
            f.write(f"Forensic Audit Report - Grade: {grade}\n")
            f.write("="*50 + "\n")
            for line in self.report:
                f.write(line + "\n")
            f.write("-" * 50 + "\n")
            f.write(f"Scores: {self.scores}\n")

if __name__ == "__main__":
    auditor = ForensicAuditor()
    auditor.run_full_audit()
