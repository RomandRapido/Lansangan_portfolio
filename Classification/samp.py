import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    """Load customer churn data"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return df

# 1. Basic Preprocessing
def preprocess_data(df):
    """Basic preprocessing steps"""
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Drop customer ID as it's not predictive
    if 'CustomerID' in data.columns:
        data.drop('CustomerID', axis=1, inplace=True)
    
    # Convert 'TotalCharges' to numeric if it's not already
    if data['TotalCharges'].dtype == 'object':
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Fill missing values with 0 or median
        data['TotalCharges'].fillna(0, inplace=True)
    
    # Convert binary categorical values to numeric (0/1)
    binary_cols = ['SeniorCitizen']
    for col in ['Gender', 'Partner', 'Dependents', 'PhoneService', 'Churn']:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
            binary_cols.append(col)
    
    # Create X (features) and y (target)
    y = data['Churn']
    X = data.drop('Churn', axis=1)
    
    return X, y, binary_cols

# 2. Feature Engineering Function
def engineer_features(X, y=None):
    """Create new features to improve model performance"""
    data = X.copy()
    
    # A. Customer Longevity Features
    
    # Tenure bins
    data['TenureBin'] = pd.cut(data['Tenure'], 
                              bins=[0, 12, 24, 36, 48, 60, float('inf')],
                              labels=[1, 2, 3, 4, 5, 6])
    data['IsNewCustomer'] = (data['Tenure'] <= 6).astype(int)
    data['IsLongTermCustomer'] = (data['Tenure'] >= 36).astype(int)
    
    # Average Monthly Spend
    data['AvgMonthlySpend'] = data['TotalCharges'] / (data['Tenure'] + 1)  # Add 1 to avoid division by zero
    
    # Spending pattern (is their monthly charge higher than their average?)
    data['SpendingPattern'] = (data['MonthlyCharges'] > data['AvgMonthlySpend']).astype(int)
    
    # B. Financial Features
    
    # Log transformation of charges (for Logistic Regression)
    data['LogMonthlyCharges'] = np.log1p(data['MonthlyCharges'])
    data['LogTotalCharges'] = np.log1p(data['TotalCharges'])
    
    # Price sensitivity flags
    month_charge_quantiles = data['MonthlyCharges'].quantile([0.25, 0.75]).values
    data['IsLowSpender'] = (data['MonthlyCharges'] <= month_charge_quantiles[0]).astype(int)
    data['IsHighSpender'] = (data['MonthlyCharges'] >= month_charge_quantiles[1]).astype(int)
    
    # C. Service Features
    
    # Create dummies for InternetService and Contract
    # These will be handled by the column transformer, but we'll create some combinations
    
    # Create service complexity score (number of services)
    # Assuming 'Yes' = 1 and 'No' = 0 for service columns after preprocessing
    service_cols = ['PhoneService']
    if 'InternetService' in data.columns:
        # For InternetService, create binary indicators
        data['HasFiberOptic'] = (data['InternetService'] == 'Fiber optic').astype(int)
        data['HasDSL'] = (data['InternetService'] == 'DSL').astype(int)
        data['HasNoInternet'] = (data['InternetService'] == 'No').astype(int)
        service_cols.extend(['HasFiberOptic', 'HasDSL'])
    
    # Contract Type
    if 'Contract' in data.columns:
        data['IsMonthToMonth'] = (data['Contract'] == 'Month-to-month').astype(int)
        data['IsOneYear'] = (data['Contract'] == 'One year').astype(int)
        data['IsTwoYear'] = (data['Contract'] == 'Two year').astype(int)
    
    # D. Interaction Features
    
    # Customer demographics + financial
    if 'SeniorCitizen' in data.columns:
        data['Senior_HighSpender'] = data['SeniorCitizen'] * data['IsHighSpender']
    
    if 'Partner' in data.columns:
        data['Partner_LongTerm'] = data['Partner'] * data['IsLongTermCustomer']
    
    # Service + Contract interaction
    if 'HasFiberOptic' in data.columns and 'IsMonthToMonth' in data.columns:
        data['FiberOptic_MonthToMonth'] = data['HasFiberOptic'] * data['IsMonthToMonth']
        # This is a high churn-risk group typically
    
    # E. Non-linear transformations (especially for Logistic Regression)
    
    # Polynomial features for Tenure
    data['Tenure_Squared'] = data['Tenure'] ** 2
    
    # Ratio features
    data['Charges_Tenure_Ratio'] = data['TotalCharges'] / (data['Tenure'] + 1)
    
    # F. Additional domain-specific features
    
    # Loyalty Index (higher for customers with long tenure and longer contracts)
    if 'IsOneYear' in data.columns and 'IsTwoYear' in data.columns:
        contract_score = data['IsMonthToMonth'] * 1 + data['IsOneYear'] * 2 + data['IsTwoYear'] * 3
        data['LoyaltyIndex'] = (data['Tenure'] / 72) * 0.5 + (contract_score / 3) * 0.5
    
    return data

# 3. Build preprocessing pipelines specific to each model
def build_preprocessors(X, binary_cols):
    """Build preprocessing pipelines for Random Forest and Logistic Regression"""
    
    # Identify column types
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object' and col not in binary_cols]
    numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64'] and col not in binary_cols]
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    print(f"Binary columns: {binary_cols}")
    
    # For Random Forest
    rf_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('bin', 'passthrough', binary_cols)
        ]
    )
    
    # For Logistic Regression
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),  # Standardize numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('bin', 'passthrough', binary_cols)
        ]
    )
    
    return rf_preprocessor, lr_preprocessor

# 4. Model Training and Evaluation
def train_and_evaluate_models(X, y, rf_preprocessor, lr_preprocessor):
    """Train and evaluate Random Forest and Logistic Regression models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Define models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    # Create pipelines
    rf_pipeline = Pipeline([
        ('preprocessor', rf_preprocessor),
        ('classifier', rf_model)
    ])
    
    lr_pipeline = Pipeline([
        ('preprocessor', lr_preprocessor),
        ('classifier', lr_model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nRandom Forest Cross-Validation:")
    rf_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"ROC AUC: {rf_scores.mean():.4f} (±{rf_scores.std():.4f})")
    
    print("\nLogistic Regression Cross-Validation:")
    lr_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"ROC AUC: {lr_scores.mean():.4f} (±{lr_scores.std():.4f})")
    
    # Train final models
    rf_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    rf_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    lr_pred_proba = lr_pipeline.predict_proba(X_test)[:, 1]
    
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    lr_auc = roc_auc_score(y_test, lr_pred_proba)
    
    print("\nTest Set Results:")
    print(f"Random Forest ROC AUC: {rf_auc:.4f}")
    print(f"Logistic Regression ROC AUC: {lr_auc:.4f}")
    
    # Feature importance for Random Forest
    if hasattr(rf_pipeline.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        
        # Get names from OneHotEncoder
        categorical_features = []
        if hasattr(rf_preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            categorical_features = rf_preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
        
        # Add numerical and binary feature names
        numeric_features = rf_preprocessor.transformers_[0][2]
        binary_features = rf_preprocessor.transformers_[2][2]
        
        feature_names = numeric_features + categorical_features + binary_features
        
        # Get feature importances
        importances = rf_pipeline.named_steps['classifier'].feature_importances_
        
        # Only print top 15 features to keep it manageable
        if len(feature_names) == len(importances):
            feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(15)
            
            print("\nTop 15 Feature Importances (Random Forest):")
            print(feature_importance)
            
    return rf_pipeline, lr_pipeline

# 5. Main function
def main():
    """Main execution function"""
    
    # 1. Load data
    df = load_data('customer_churn.csv')
    
    # 2. Basic preprocessing
    X, y, binary_cols = preprocess_data(df)
    
    # 3. Engineer features
    X_engineered = engineer_features(X, y)
    
    # 4. Build preprocessors
    rf_preprocessor, lr_preprocessor = build_preprocessors(X_engineered, binary_cols)
    
    # 5. Train and evaluate models
    rf_model, lr_model = train_and_evaluate_models(X_engineered, y, rf_preprocessor, lr_preprocessor)
    
    print("\nFeature engineering pipeline completed successfully!")
    
if __name__ == "__main__":
    main()