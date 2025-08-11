# NuroSight - Business Analytics Platform

NuroSight is a full-stack business analytics web application that provides automated data analysis, machine learning model training, and prediction capabilities.

## Features

- File upload support for CSV and Excel files
- Automated data cleaning and preprocessing
- Multiple machine learning models (Linear Regression, Random Forest, XGBoost)
- Interactive visualizations with Chart.js
- Real-time predictions
- Comprehensive analytics dashboard
- Mobile-responsive design

## Technology Stack

- Backend: Python (Flask)
- Frontend: HTML + TailwindCSS + Chart.js
- Machine Learning: pandas, numpy, scikit-learn, xgboost

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nurosight
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the values as needed

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Upload Data**
   - Go to the home page
   - Upload your CSV or Excel file
   - The system will automatically clean and process your data

2. **Train Model**
   - Select your target variable
   - Choose features for prediction
   - Train multiple models and compare their performance

3. **Make Predictions**
   - Upload new data for batch predictions
   - Or use the web form for single predictions
   - Download results in CSV format

4. **View Analytics**
   - Access the dashboard for detailed insights
   - View correlation heatmaps
   - Analyze feature importance
   - Track prediction trends

## Project Structure

```
nurosight/
├── app.py              # Main Flask application
├── ml_utils.py         # Machine learning utilities
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── train.html
│   ├── predict.html
│   └── dashboard.html
├── models/            # Saved ML models
└── uploads/          # Temporary file storage
```

## API Endpoints

- `POST /upload` - Handle file uploads
- `POST /train` - Train ML models
- `POST /predict` - Make predictions
- `GET /metrics` - Get model metrics

## Development

1. Create a new branch for features
2. Follow PEP 8 style guidelines
3. Add tests for new features
4. Submit pull requests

## Deployment

The application is ready for deployment on platforms like Render or Heroku. Make sure to:

1. Set up environment variables
2. Configure the web server (Gunicorn)
3. Set up database if needed
4. Configure CORS and security settings

## License

[MIT License](LICENSE)

## Support

For support, please raise an issue in the GitHub repository.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
