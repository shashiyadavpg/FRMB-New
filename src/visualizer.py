import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
import os

class Visualizer:
    def __init__(self, data_path='outputs/model_output_scored.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        
    def generate_plots(self):
        print("Generating plots...")
        if not os.path.exists(self.data_path):
            print("Model output not found.")
            return

        df = pd.read_csv(self.data_path)
        
        # Ensure we have target
        target = 'default_within_1y'
        if target not in df.columns:
            print("Target not in dataset, cannot plot ROC/Calibration.")
            return
            
        # Filter for Test set (Year >= 2022) to show validation performance
        # Or just use the whole set with a note? Usually validation plots are on Test set.
        # But here 'model_output_scored.csv' has correct predictions? 
        # Actually 'PD_XGB' column in scored file was trained on Train and predicted on Test? 
        # In 'train_models', I predicted on ALL data. So for fair evaluation plots, I should filter for Test years.
        
        test_df = df[df['Year'] >= 2022]
        
        if test_df.empty:
            print("No test data found for plotting.")
            return
            
        y_true = test_df[target]
        y_prob = test_df['PD_XGB'] # Using XGB as primary
        
        # 1. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label='XGBoost')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Test Set)')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
        
        # 2. Calibration Plot
        plt.figure(figsize=(8, 6))
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'calibration_curve.png'))
        plt.close()
        
        # 3. PD Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['PD_XGB'], bins=50, kde=True)
        plt.title('Distribution of PD Estimates (All Companies)')
        plt.xlabel('Probability of Default')
        plt.savefig(os.path.join(self.output_dir, 'pd_distribution.png'))
        plt.close()
        
        # 4. Top Risky Companies (Bar)
        max_year = df['Year'].max()
        latest = df[df['Year'] == max_year].sort_values('PD_XGB', ascending=False).head(15)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='PD_XGB', y='Ticker', data=latest, palette='Reds_r')
        plt.title(f'Top 15 Risky Companies ({max_year})')
        plt.savefig(os.path.join(self.output_dir, 'top_risky_bar.png'))
        plt.close()

if __name__ == "__main__":
    viz = Visualizer()
    viz.generate_plots()
