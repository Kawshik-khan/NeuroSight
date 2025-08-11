import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib

def read_csv_with_fallback(filepath):
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
            continue
    # Last resort: replace errors
    try:
        return pd.read_csv(
            filepath,
            encoding='latin1',
            encoding_errors='replace',
            low_memory=False
        )
    except Exception:
        if last_error is not None:
            raise last_error
        raise ValueError('Failed to read CSV with common encodings')

def load_and_clean_data(filepath):
    # Load data based on file extension
    if filepath.endswith('.csv'):
        df = read_csv_with_fallback(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Handle missing values
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Handle outliers using IQR method for numeric columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def train_models(df, target_column, features=None):
    if features is None:
        features = [col for col in df.columns if col != target_column]
    
    # Prepare data
    X = df[features]
    y = df[target_column]

    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors='coerce')
    # Drop rows where target could not be converted
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    if len(y) < 2:
        raise ValueError('Not enough valid rows after cleaning to train (need at least 2).')
    
    # Convert categorical variables to numeric
    X = pd.get_dummies(X)
    
    # Split data with safe test size (at least 1 sample in test and train)
    n_samples = len(X)
    test_size = max(1, int(round(0.2 * n_samples)))
    test_size = min(test_size, n_samples - 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(random_state=42)
    
    results = {}
    best_score = -float('inf')
    best_model = None
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
    
    results['best_model'] = best_model
    results['features'] = X.columns.tolist()
    
    return results

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def make_prediction(model, df):
    # Ensure the dataframe has the same features as training data
    features = joblib.load('models/features.joblib')
    df = pd.get_dummies(df)
    
    # Add missing columns
    for col in features:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[features]
    
    return model.predict(df)

def get_data_summary(df):
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Calculate correlations for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        correlations = df[numeric_cols].corr().to_dict()
        summary['correlations'] = correlations
    
    return summary
