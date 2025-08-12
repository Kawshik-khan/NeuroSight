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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

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

def fix_data_types(df):
    """Fix data types for common columns that should be numeric or datetime"""
    df_fixed = df.copy()
    
    # Common date columns that should be converted
    date_columns = ['date', 'Date', 'DATE', 'order_date', 'Order_Date', 'ORDER_DATE', 
                    'purchase_date', 'Purchase_Date', 'PURCHASE_DATE', 'transaction_date',
                    'Transaction_Date', 'TRANSACTION_DATE', 'created_at', 'Created_At',
                    'timestamp', 'Timestamp', 'TIMESTAMP']
    
    # Common numeric columns that might be strings
    numeric_columns = ['amount', 'Amount', 'AMOUNT', 'price', 'Price', 'PRICE',
                      'quantity', 'Quantity', 'QUANTITY', 'total', 'Total', 'TOTAL',
                      'revenue', 'Revenue', 'REVENUE', 'sales', 'Sales', 'SALES',
                      'cost', 'Cost', 'COST', 'profit', 'Profit', 'PROFIT']
    
    # Convert date columns
    for col in df_fixed.columns:
        if col.lower() in [d.lower() for d in date_columns]:
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
            except:
                pass
    
    # Convert numeric columns
    for col in df_fixed.columns:
        if col.lower() in [n.lower() for n in numeric_columns]:
            try:
                df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
            except:
                pass
    
    return df_fixed

def perform_rfqu_analysis(df, customer_id_col, date_col, unitprice_col, quantity_col, n_clusters=3):
    """
    Perform RFQU (Recency, Frequency, Quantity, UnitPrice, Monetary) analysis with K-means clustering
    """
    # Validate required columns
    required_cols = [customer_id_col, date_col, unitprice_col, quantity_col]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create a copy to avoid modifying original data
    df_fixed = df.copy()
    
    # Convert date column to datetime
    try:
        df_fixed[date_col] = pd.to_datetime(df_fixed[date_col])
    except Exception as e:
        raise ValueError(f"Error converting {date_col} to datetime: {str(e)}")
    
    # Convert unitprice and quantity to numeric
    try:
        df_fixed[unitprice_col] = pd.to_numeric(df_fixed[unitprice_col], errors='coerce')
        df_fixed[quantity_col] = pd.to_numeric(df_fixed[quantity_col], errors='coerce')
    except Exception as e:
        raise ValueError(f"Error converting {unitprice_col} or {quantity_col} to numeric: {str(e)}")
    
    # Remove rows with missing values in required columns
    df_fixed = df_fixed.dropna(subset=[customer_id_col, date_col, unitprice_col, quantity_col])
    
    if df_fixed.empty:
        raise ValueError("No valid data remaining after cleaning")
    
    # Calculate RFQU metrics
    current_date = df_fixed[date_col].max()
    
    # Calculate monetary value for each transaction first
    df_fixed['monetary'] = df_fixed[quantity_col] * df_fixed[unitprice_col]
    
    rfqu = df_fixed.groupby(customer_id_col).agg({
        date_col: lambda x: (current_date - x.max()).days,  # Recency
        customer_id_col: 'nunique',  # Frequency
        quantity_col: 'sum',  # Total Quantity
        unitprice_col: 'mean',  # Average Unit Price
        'monetary': 'sum'  # Total Monetary Value
    }).rename(columns={
        date_col: 'recency',
        customer_id_col: 'frequency',
        quantity_col: 'quantity',
        unitprice_col: 'unitprice'
    })
    
    # Calculate RFQU scores (1-5 scale, 5 being best)
    # Use tie-safe ranking before qcut to avoid duplicate bin edges
    def _quantile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
        ranked = values.rank(method='first', ascending=True)
        unique_ranks = ranked.nunique()
        bins = int(min(5, unique_ranks))
        if bins < 2:
            # Not enough variation; assign neutral score
            return pd.Series(np.full(len(ranked), 3, dtype=int), index=ranked.index)
        labels = list(range(1, bins + 1)) if higher_is_better else list(range(bins, 0, -1))
        return pd.qcut(ranked, q=bins, labels=labels)

    rfqu['r_score'] = _quantile_score(rfqu['recency'], higher_is_better=False)
    rfqu['f_score'] = _quantile_score(rfqu['frequency'], higher_is_better=True)
    rfqu['q_score'] = _quantile_score(rfqu['quantity'], higher_is_better=True)
    rfqu['u_score'] = _quantile_score(rfqu['unitprice'], higher_is_better=True)
    rfqu['m_score'] = _quantile_score(rfqu['monetary'], higher_is_better=True) # Add Monetary score
    
    # Convert scores to numeric
    rfqu['r_score'] = pd.to_numeric(rfqu['r_score'])
    rfqu['f_score'] = pd.to_numeric(rfqu['f_score'])
    rfqu['q_score'] = pd.to_numeric(rfqu['q_score'])
    rfqu['u_score'] = pd.to_numeric(rfqu['u_score'])
    rfqu['m_score'] = pd.to_numeric(rfqu['m_score']) # Convert Monetary score
    
    # Prepare data for clustering
    rfqu_features = rfqu[['r_score', 'f_score', 'q_score', 'u_score', 'm_score']].values
    
    # Scale the features
    scaler = StandardScaler()
    rfqu_scaled = scaler.fit_transform(rfqu_features)
    
    # Perform K-means clustering
    try:
        print(f"Using {n_clusters} clusters for K-means clustering")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rfqu_scaled)
        print(f"K-means clustering completed with {len(np.unique(cluster_labels))} unique clusters")
    except Exception as e:
        raise ValueError(f"K-means clustering failed: {str(e)}. Try reducing the number of clusters or check your data.")
    
    # Add cluster labels to RFQU dataframe
    rfqu['cluster'] = cluster_labels
    
    # Create segment names based on cluster characteristics
    segment_names = []
    for i in range(n_clusters):
        cluster_data = rfqu[rfqu['cluster'] == i]
        avg_r = cluster_data['r_score'].mean()
        avg_f = cluster_data['f_score'].mean()
        avg_m = cluster_data['m_score'].mean()
        
        # Simple 3-segment logic for RFQU analysis
        if avg_r >= 4 and avg_f >= 3:
            # High recency (recent) and good frequency
            segment = "Active Customers"
        elif avg_r >= 3 and avg_f >= 3:
            # Medium recency and frequency
            segment = "Regular Customers"
        else:
            # Low recency or frequency
            segment = "At Risk Customers"
        
        segment_names.append(segment)
    
    # Map segment names to clusters
    rfqu['segment'] = rfqu['cluster'].map(dict(enumerate(segment_names)))
    
    # Calculate cluster statistics
    cluster_stats = rfqu.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'quantity': 'mean',
        'unitprice': 'mean',
        'monetary': 'mean', # Add Monetary to stats
        'r_score': 'mean',
        'f_score': 'mean',
        'q_score': 'mean',
        'u_score': 'mean',
        'm_score': 'mean' # Add Monetary to stats
    }).round(2)
    
    cluster_stats['segment'] = segment_names
    cluster_stats['count'] = rfqu.groupby('cluster').size()
    
    return {
        'rfqu_data': rfqu,
        'cluster_stats': cluster_stats,
        'cluster_centers': kmeans.cluster_centers_,
        'segment_names': segment_names,
        'scaler': scaler,
        'kmeans_model': kmeans
    }

def save_rfqu_model(rfqu_results, filepath):
    """Save RFQU analysis results and model"""
    # Remove non-serializable objects before saving
    rfqu_data = rfqu_results.copy()
    if 'scaler' in rfqu_data:
        del rfqu_data['scaler']
    if 'kmeans_model' in rfqu_data:
        del rfqu_data['kmeans_model']
    
    joblib.dump(rfqu_data, filepath)

def load_rfqu_model(filepath):
    """Load RFQU analysis results"""
    return joblib.load(filepath)
