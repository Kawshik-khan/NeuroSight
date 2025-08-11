from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastMCP Server
app = FastMCP("NuroSight")
logging.info("FastMCP Server initialized")

# Data preprocessing functions
def clean_data(df):
    """Clean and preprocess the data"""
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

@app.tool()
def load_data(file_path: str) -> dict:
    """Load and analyze data from CSV or Excel file"""
    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        df = clean_data(df)
        
        return {
            "status": "success",
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.tool()
def train_model(
    file_path: str,
    target_column: str,
    features: list = None,
    model_type: str = "random_forest"
) -> dict:
    """Train a machine learning model"""
    try:
        # Load and prepare data
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        df = clean_data(df)
        
        if features is None:
            features = [col for col in df.columns if col != target_column]
        
        X = df[features]
        y = df[target_column]
        
        # Convert categorical variables to numeric
        X = pd.get_dummies(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select and train model
        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "xgboost": XGBRegressor(random_state=42)
        }
        
        model = models.get(model_type, models["random_forest"])
        model.fit(X_train, y_train)
        
        # Get predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            "status": "success",
            "model_type": model_type,
            "metrics": {
                "train_r2": r2_score(y_train, train_pred),
                "test_r2": r2_score(y_test, test_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
                "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred))
            },
            "feature_importance": None
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = dict(zip(X.columns, model.feature_importances_))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_type}_model.joblib'
        joblib.dump(model, model_path)
        
        # Save feature names for prediction
        joblib.dump(X.columns.tolist(), f'models/{model_type}_features.joblib')
        
        return results
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.tool()
def predict(
    input_data: dict,
    model_type: str = "random_forest"
) -> dict:
    """Make predictions using a trained model"""
    try:
        # Load model and features
        model_path = f'models/{model_type}_model.joblib'
        features_path = f'models/{model_type}_features.joblib'
        
        if not (os.path.exists(model_path) and os.path.exists(features_path)):
            return {
                "status": "error",
                "message": f"Model {model_type} not found. Please train the model first."
            }
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        
        # Prepare input data
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df)
        
        # Add missing columns
        for col in features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[features]
        
        # Make prediction
        prediction = model.predict(df)
        
        return {
            "status": "success",
            "prediction": prediction.tolist()[0]
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.tool()
def batch_predict(
    file_path: str,
    model_type: str = "random_forest"
) -> dict:
    """Make predictions for a batch of data"""
    try:
        # Load model and features
        model_path = f'models/{model_type}_model.joblib'
        features_path = f'models/{model_type}_features.joblib'
        
        if not (os.path.exists(model_path) and os.path.exists(features_path)):
            return {
                "status": "error",
                "message": f"Model {model_type} not found. Please train the model first."
            }
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        
        # Load and prepare input data
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        df = clean_data(df)
        df = pd.get_dummies(df)
        
        # Add missing columns
        for col in features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[features]
        
        # Make predictions
        predictions = model.predict(df)
        
        return {
            "status": "success",
            "predictions": predictions.tolist()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    app.run()
