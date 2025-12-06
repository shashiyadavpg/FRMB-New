import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, roc_curve, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class AdvancedPDModelTrainer:
    def __init__(self, data_path='data/processed/model_input.csv', output_dir='outputs'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.features = [
            'ROA', 'DebtToEquity', 'NetMargin', 'EBITDAMargin', 
            'DebtToAssets', 'InterestCoverage', 
            'OCF_to_Debt', 'CurrentRatio', 'MarketCap_to_Assets',
            'Return_1Y', 'Volatility_1Y', 'Z_Score', 'O_Score',
            'ROA_Lag1', 'DE_Lag1', 'Slope_Leverage'
        ]
        self.target = 'default_within_1y'

    def check_vif(self, X):
        print("\n[FORENSIC] Checking VIF (Multicollinearity)...")
        # Handle Inf/Nan
        X_clean = X.replace([np.inf, -np.inf], 0).fillna(0)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_clean.columns
        vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(len(X_clean.columns))]
        
        print(vif_data.sort_values('VIF', ascending=False))
        vif_data.to_csv(os.path.join(self.output_dir, 'vif_analysis.csv'), index=False)
        return vif_data

    def load_and_split(self):
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.features + [self.target])
        
        # YEAR-BASED SPLIT (Correct methodology to prevent lookahead)
        # Train: < 2023
        # Test: >= 2023
        cutoff_year = 2023
        train = df[df['Year'] < cutoff_year]
        test = df[df['Year'] >= cutoff_year]
        
        # Check if we have defaults in both sets
        print(f"Train Defaults: {train[self.target].sum()} / {len(train)}")
        print(f"Test Defaults: {test[self.target].sum()} / {len(test)}")
        
        if test[self.target].sum() == 0:
            print("WARNING: No defaults in test set (2023+). Relaxing cutoff to 2022.")
            cutoff_year = 2022
            train = df[df['Year'] < cutoff_year]
            test = df[df['Year'] >= cutoff_year]
            print(f"New Train Defaults: {train[self.target].sum()}")
            print(f"New Test Defaults: {test[self.target].sum()}")

        return train, test, df

    def train_models(self):
        train, test, full_df = self.load_and_split()
        
        X_train = train[self.features]
        y_train = train[self.target]
        X_test = test[self.features]
        y_test = test[self.target]
        
        # VIF Check
        self.check_vif(X_train)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        models = {}
        
        # 1. Logistic Regression (Whitebox)
        print("Training Logistic Regression...")
        lr = LogisticRegression(penalty='l2', C=0.5, class_weight='balanced', max_iter=2000)
        lr.fit(X_train_scaled, y_train)
        models['Logistic'] = lr
        
        pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
        results['Logistic'] = self.evaluate(y_test, pred_lr)
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        pred_rf = rf.predict_proba(X_test)[:, 1]
        results['RandomForest'] = self.evaluate(y_test, pred_rf)
        
        # 3. XGBoost (Likely best performer)
        print("Training XGBoost...")
        # Monotonic constraints could be added here for "Economic Sense"
        xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                      scale_pos_weight=10, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results['XGBoost'] = self.evaluate(y_test, pred_xgb)
        
        # 4. Calibrated XGBoost (Isotonic)
        print("Calibrating XGBoost...")
        cal_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
        cal_xgb.fit(X_test, y_test) # Ideally should use validation set, but scarce data. Using Test for demo calibration correction.
        # Actually, let's just create a calibrated version using the training set CV?
        # Standard way: CalibratedClassifierCV(base_estimator, cv=3) on Train.
        cal_xgb_train = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
        cal_xgb_train.fit(X_train, y_train)
        models['XGB_Calibrated'] = cal_xgb_train
        pred_cal_xgb = cal_xgb_train.predict_proba(X_test)[:, 1]
        results['XGB_Calibrated'] = self.evaluate(y_test, pred_cal_xgb)

        # Save Results
        self.save_artifacts(results, models, X_train.columns)
        
        # Full Scoring
        X_full = full_df[self.features]
        X_full_scaled = scaler.transform(X_full)
        
        full_df['PD_Logistic'] = lr.predict_proba(X_full_scaled)[:, 1]
        full_df['PD_XGB'] = xgb_model.predict_proba(X_full)[:, 1]
        full_df['PD_Calibrated'] = cal_xgb_train.predict_proba(X_full)[:, 1]
        
        full_df.to_csv(os.path.join(self.output_dir, 'model_output_premium.csv'), index=False)
        print("Saved model_output_premium.csv")
        
        return results

    def evaluate(self, y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
        
        # Precision @ Top 20 (Simulated)
        # Sort by pred, check if actual default in top k
        rank_df = pd.DataFrame({'true': y_true, 'pred': y_pred}).sort_values('pred', ascending=False)
        top_20_defaults = rank_df.head(20)['true'].sum()
        prec_top20 = top_20_defaults / 20.0 if len(rank_df) >= 20 else top_20_defaults / len(rank_df)
        
        return {
            'AUC': auc,
            'KS': ks,
            'LogLoss': logloss,
            'Brier': brier,
            'Precision@Top20': prec_top20
        }

    def save_artifacts(self, results, models, feature_names):
        print(json.dumps(results, indent=2))
        with open(os.path.join(self.output_dir, 'premium_metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # Feature Importance (SHAP-like via Gain)
        xgb_imp = models['XGBoost'].feature_importances_
        pd.DataFrame({'Feature': feature_names, 'Importance': xgb_imp})\
            .sort_values('Importance', ascending=False)\
            .to_csv(os.path.join(self.output_dir, 'premium_features.csv'), index=False)
            
        # Logistic Coefs
        lr_coef = models['Logistic'].coef_[0]
        pd.DataFrame({'Feature': feature_names, 'Coefficient': lr_coef})\
            .sort_values('Coefficient', key=abs, ascending=False)\
            .to_csv(os.path.join(self.output_dir, 'premium_coefficients.csv'), index=False)

if __name__ == "__main__":
    trainer = AdvancedPDModelTrainer()
    trainer.train_models()
