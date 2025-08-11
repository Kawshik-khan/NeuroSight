from flask import Flask, request, render_template, redirect, flash, url_for, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Create models directory if it doesn't exist
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_clean_data(filepath):
    """Load and preprocess the dataset."""
    # Load the data
    if filepath.lower().endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    # Basic cleaning
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values - for now, we'll drop rows with any missing values
    df = df.dropna()
    
    return df

def train_models(df, target_column, feature_columns):
    """Train multiple models and return the best one."""
    # Prepare the data
    X = df[feature_columns]
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
    
    # Store scaler and feature columns with the model
    model_info = {
        'model': rf_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'metrics': {
            'mse': float(mse),
            'r2': float(r2),
            'rmse': float(np.sqrt(mse))
        },
        'feature_importance': feature_importance
    }
    
    return model_info

def save_model(model_info, filepath):
    """Save the model and associated information."""
    joblib.dump(model_info, filepath)

def load_model(filepath):
    """Load the model and associated information."""
    return joblib.load(filepath)

def make_prediction(model_info, df):
    """Make predictions using the trained model."""
    # Ensure we have all required features
    missing_features = set(model_info['feature_columns']) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    # Extract features in the correct order
    X = df[model_info['feature_columns']]
    
    # Scale features
    X_scaled = model_info['scaler'].transform(X)
    
    # Make predictions
    predictions = model_info['model'].predict(X_scaled)
    
    return predictions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        if not allowed_file(file.filename):
            return {'error': 'Invalid file type. Please upload a CSV or Excel file.'}, 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # Save the file
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Read the file to validate and get columns
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Verify the file was read successfully
            if df.empty:
                return {'error': 'The uploaded file is empty'}, 400
            
            # Store file info in session
            columns = df.columns.tolist()
            
            # Return success response with next steps
            return {
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'columns': columns,
                'nextUrl': url_for('train')  # URL for the training page
            }, 200
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            print(traceback.format_exc())
            return {'error': f'Error processing file: {str(e)}'}, 500
            
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return {'error': 'An unexpected error occurred'}, 500

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get request data
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        target_column = data.get('target')
        if not target_column:
            return jsonify({'error': 'No target variable specified'}), 400
            
        features = data.get('features', [])
        if not features:
            return jsonify({'error': 'No features specified'}), 400
            
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'No file specified'}), 400
        
        # Load and process the data
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        try:
            df = load_and_clean_data(filepath)
        except Exception as e:
            return jsonify({'error': f'Error loading data: {str(e)}'}), 400
        
        # Validate columns
        missing_columns = set([target_column] + features) - set(df.columns)
        if missing_columns:
            return jsonify({'error': f'Missing columns in dataset: {missing_columns}'}), 400
        
        # Train the model
        try:
            model_info = train_models(df, target_column, features)
        except Exception as e:
            return jsonify({'error': f'Error during training: {str(e)}'}), 500
        
        # Save the model
        model_path = os.path.join(MODELS_FOLDER, 'model.joblib')
        try:
            save_model(model_info, model_path)
        except Exception as e:
            return jsonify({'error': f'Error saving model: {str(e)}'}), 500
        
        # Return results
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': model_info['metrics'],
            'feature_importance': model_info['feature_importance']
        })
        
    except Exception as e:
        print(f"Unexpected error during training: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def make_predictions():
    try:
        # Load the model
        model_path = os.path.join(MODELS_FOLDER, 'model.joblib')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
            
        try:
            model_info = load_model(model_path)
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
        
        # Handle input data
        try:
            if 'file' in request.files:
                # Batch predictions
                file = request.files['file']
                if not allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
                    
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                df = load_and_clean_data(filepath)
            else:
                # Single prediction
                data = request.json
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                df = pd.DataFrame([data])
            
            # Make predictions
            predictions = make_prediction(model_info, df)
            
            return jsonify({
                'success': True,
                'predictions': predictions.tolist()
            })
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Error making predictions: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Unexpected error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/metrics')
def get_metrics():
    try:
        # Load the model
        model_path = os.path.join(MODELS_FOLDER, 'model.joblib')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
            
        try:
            model_info = load_model(model_path)
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
        
        # Return model metrics and information
        return jsonify({
            'success': True,
            'model_type': str(type(model_info['model']).__name__),
            'features': model_info['feature_columns'],
            'metrics': model_info['metrics'],
            'feature_importance': model_info['feature_importance']
        })
        
    except Exception as e:
        print(f"Unexpected error getting metrics: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=os.getenv('DEBUG', 'False').lower() == 'true')
