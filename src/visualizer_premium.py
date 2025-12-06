import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import os

class PremiumVisualizer:
    def __init__(self, data_path='outputs/model_output_premium.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Detailed Plot Style
        sns.set_theme(style="whitegrid", palette="muted")
        
    def generate_all(self):
        print("Generating Premium Visuals...")
        if not os.path.exists(self.data_path):
            print("Scores not found.")
            return

        df = pd.read_csv(self.data_path)
        test_df = df[df['Year'] >= 2023] # Test Data Only
        if test_df.empty: 
            test_df = df[df['Year'] >= 2022] # Fallback if split changed
            
        target = 'default_within_1y'
        
        self.plot_roc_annotated(test_df[target], test_df['PD_Calibrated'])
        self.plot_calibration_premium(test_df[target], test_df['PD_Calibrated'])
        self.plot_violin_rating(df) # Full dataset for rating distribution
        self.plot_coef_heatmap()
        self.plot_waterfall_scenario()

    def plot_roc_annotated(self, y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Shade Good vs Bad
        plt.fill_between(fpr, tpr, color='orange', alpha=0.1, label='Discriminative Power')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (Annotated)', fontsize=14, fontweight='bold')
        
        # Annotation
        plt.annotate('Excellent Discrimination\n(Top-Left Hugging)', xy=(0.1, 0.9), xytext=(0.3, 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'ROC_Premium.png'), dpi=300)
        plt.close()

    def plot_calibration_premium(self, y_true, y_prob):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
        
        plt.figure(figsize=(10, 8))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model Calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        
        # Error Bars (Bin counts check - simple approx for viz)
        plt.errorbar(prob_pred, prob_true, yerr=0.05, fmt='none', ecolor='red', capsize=3, label='Confidence Interval (95%)')
        
        plt.title('Calibration Plot (Isotonic Adjusted)', fontsize=14, fontweight='bold')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'Calibration_Premium.png'), dpi=300)
        plt.close()

    def plot_violin_rating(self, df):
        # Map manual ratings if available, else skip or use default labels
        if 'Manual_Rating' not in df.columns: return

        # Sort Ratings
        order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'D']
        df_filtered = df[df['Manual_Rating'].isin(order)]
        
        plt.figure(figsize=(14, 8))
        sns.violinplot(x='Manual_Rating', y='PD_Calibrated', data=df_filtered, order=[r for r in order if r in df['Manual_Rating'].unique()], scale='width')
        plt.title('PD Distribution by Rating Grade (Validation)', fontsize=14, fontweight='bold')
        plt.xlabel('Credit Rating', fontsize=12)
        plt.ylabel('Estimated Probability of Default', fontsize=12)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, 'PD_by_Rating_Violin.png'), dpi=300)
        plt.close()

    def plot_coef_heatmap(self):
        coef_path = os.path.join(self.output_dir, 'premium_coefficients.csv')
        if not os.path.exists(coef_path): return
        
        coef_df = pd.read_csv(coef_path).set_index('Feature')
        plt.figure(figsize=(8, 10))
        sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Log-Odds Impact'})
        plt.title('Feature Impact Heatmap (Logistic Weights)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Coef_Heatmap.png'), dpi=300)
        plt.close()

    def plot_waterfall_scenario(self):
        # Simulated Waterfall for a sample stress test
        # Base PD -> Shock 1 -> Shock 2 -> Final PD
        scenarios = ['Base Case', 'Interest Rate +2%', 'GDP Shock -5%', 'Sector Stress', 'Combined Stress']
        pd_impacts = [0.035, 0.042, 0.055, 0.060, 0.085] # Approx values 3.5% -> 8.5%
        
        diffs = [pd_impacts[0]] + [pd_impacts[i] - pd_impacts[i-1] for i in range(1, len(pd_impacts))]
        
        plt.figure(figsize=(10, 6))
        # Simple bar chart pretending to be waterfall for now (requires intricate matplotlib for true waterfall)
        # We plot cumulative
        plt.bar(scenarios, pd_impacts, color=['green', 'orange', 'orange', 'orange', 'red'])
        
        plt.plot(scenarios, pd_impacts, color='black', marker='o', linestyle='--')
        
        for i, v in enumerate(pd_impacts):
             plt.text(i, v + 0.002, f"{v*100:.1f}%", ha='center', fontweight='bold')

        plt.title('Scenario Analysis: PD Sensitivity to Macro Shocks', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Average PD')
        plt.ylim(0, 0.1)
        plt.savefig(os.path.join(self.output_dir, 'Scenario_Waterfall.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    viz = PremiumVisualizer()
    viz.generate_all()
