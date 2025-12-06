import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class PDModelTrainer:
    def __init__(self, data_path='data/processed/model_input.csv', output_dir='outputs/'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.features = [
            'ROA', 'ROE', 'NetMargin', 'EBITDAMargin', 
            'DebtToEquity', 'DebtToAssets', 'InterestCoverage', 
            'OCF_to_Debt', 'CurrentRatio', 'MarketCap_to_Assets',
            'Return_1Y', 'Volatility_1Y', 'Z_Score', 'O_Score',
            'ROA_Lag1', 'DE_Lag1', 'Slope_Leverage'
        ]
        self.target = 'default_within_1y'

    def load_and_split(self):
        print("Loading data for modelling...")
        df = pd.read_csv(self.data_path)
        
        # Prepare Data
        # Drop Infs and NAs
        df = df.replace([np.inf, -np.inf], np.nan)
        df_clean = df.dropna(subset=self.features + [self.target])
        
        print(f"Data shape after cleaning: {df_clean.shape}")
        
        from sklearn.model_selection import train_test_split
        
        # Stratified Split to ensure defaults in both sets
        # Given the low number of defaults (historical data issues), this is necessary.
        print(f"Total Defaults in dataset: {sum(df_clean[self.target])}")
        
        if sum(df_clean[self.target]) < 5:
            print("WARNING: Very few default events found. Model results may be unstable.")
            
        X = df_clean
        y = df_clean[self.target]
        
        train, test = train_test_split(df_clean, test_size=0.3, random_state=42, stratify=y)
        
        print(f"Train set: {len(train)} rows ({train[self.target].sum()} defaults)")
        print(f"Test set: {len(test)} rows ({test[self.target].sum()} defaults)")
        
        return train, test, df_clean

    def train_models(self):
        train, test, full_df = self.load_and_split()
        
        X_train = train[self.features]
        y_train = train[self.target]
        X_test = test[self.features]
        y_test = test[self.target]
        
        # Scale for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        models = {}
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
        lr.fit(X_train_scaled, y_train)
        models['Logistic'] = lr
        
        pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
        results['Logistic'] = self.evaluate(y_test, pred_lr)
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        
        pred_rf = rf.predict_proba(X_test)[:, 1]
        results['RandomForest'] = self.evaluate(y_test, pred_rf)
        
        # 3. XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, scale_pos_weight=10) # Simple weighting
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results['XGBoost'] = self.evaluate(y_test, pred_xgb)
        
        # Save Outputs
        self.save_results(results, models, X_train.columns)
        
        # Save Predictions to CSV (for Excel)
        # Apply strict cleaning again to full df to match rows? 
        # Easier to just predict on full_df filtered
        
        # Predict on ALL data for final Dashboard
        X_full = full_df[self.features]
        X_full_scaled = scaler.transform(X_full)
        
        full_df['PD_Logistic'] = lr.predict_proba(X_full_scaled)[:, 1]
        full_df['PD_RF'] = rf.predict_proba(X_full)[:, 1]
        full_df['PD_XGB'] = xgb_model.predict_proba(X_full)[:, 1]
        
        full_df.to_csv(os.path.join(self.output_dir, 'model_output_scored.csv'), index=False)
        print("Saved scored dataset.")
        
        return results

    def evaluate(self, y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        
        # KS Statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
        
        return {
            'AUC': auc,
            'KS': ks,
            'LogLoss': logloss,
            'Brier': brier
        }

    def save_results(self, results, models, feature_names):
        # Save metrics
        print("Model Results:")
        print(json.dumps(results, indent=2))
        
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # Feature Importance (XGB)
        xgb_imp = models['XGBoost'].feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': xgb_imp}).sort_values('Importance', ascending=False)
        imp_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        
        # Coefficients (Logistic)
        lr_coef = models['Logistic'].coef_[0]
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lr_coef}).sort_values('Coefficient', key=abs, ascending=False)
        coef_df.to_csv(os.path.join(self.output_dir, 'logistic_coefficients.csv'), index=False)

if __name__ == "__main__":
    trainer = PDModelTrainer()
    trainer.train_models()
