# Loan Approval Predictor

A machine learning-powered web application that predicts loan approval decisions based on applicant information and loan details.

## What This Project Does

This Flask-based web application uses a trained machine learning model to predict whether a loan application will be **approved** or **rejected**. The prediction is based on various factors including:

- **Personal Information**: Age, gender, education level, income, employment experience
- **Loan Details**: Loan amount, interest rate, loan-to-income ratio
- **Credit History**: Credit score, credit history length, previous loan defaults
- **Property Ownership**: Home ownership status
- **Loan Purpose**: Intent of the loan (personal, education, medical, etc.)

The application provides:
- ✅ **Real-time predictions** with probability scores
- 🔒 **Input validation** and security measures
- 📊 **Rate limiting** to prevent abuse
- 📝 **Comprehensive logging** for monitoring
- 🎨 **Modern web interface** with responsive design

## Features

### Core Functionality
- **Machine Learning Prediction**: Uses a trained classifier model to predict loan outcomes
- **Web Interface**: Clean, modern UI built with Flask and Tailwind CSS
- **Data Validation**: Comprehensive input validation with helpful error messages
- **Security**: Rate limiting, security headers, and input sanitization

### Technical Features
- **Model Loading**: Automatically loads pre-trained models on startup
- **Data Preprocessing**: Handles categorical encoding and feature scaling
- **Error Handling**: Graceful error handling with user-friendly messages
- **Health Checks**: Built-in health check endpoint for monitoring
- **Logging**: Detailed logging for debugging and monitoring

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Required Python packages (see requirements.txt)

### Installation Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**
   The application requires these model files in the same directory:
   - `model.pkl` - The trained classification model
   - `scaler.pkl` - The feature scaler
   - `X_train_columns.pkl` - Column information for feature alignment

### Running the Application

#### Development Mode
```bash
python app.py
```
The application will start on `http://localhost:5000`

#### Production Mode
Set environment variables for production:
```bash
export FLASK_DEBUG=false
export SECRET_KEY="your-secret-key-here"
export PORT=5000
export RATE_LIMIT_REQUESTS=10
export RATE_LIMIT_WINDOW=60

# Using gunicorn (recommended for production)
gunicorn --bind 0.0.0.0:5000 app:app
```

## Usage

1. **Access the web interface** at `http://localhost:5000`
2. **Fill out the loan application form** with applicant details:
   - Personal information (age, gender, education, income, etc.)
   - Loan specifics (amount, interest rate, purpose)
   - Credit information (credit score, history length, defaults)
3. **Submit the form** to get an instant prediction
4. **View results**: The application will display "Approved ✅" or "Rejected ❌" along with a probability score

## API Endpoints

- `GET/POST /` - Main application interface
- `GET /health` - Health check endpoint for monitoring

## Configuration

### Environment Variables
- `FLASK_DEBUG` - Enable/disable debug mode (default: false)
- `SECRET_KEY` - Flask secret key for sessions
- `PORT` - Port to run the application on (default: 5000)
- `RATE_LIMIT_REQUESTS` - Max requests per IP (default: 10)
- `RATE_LIMIT_WINDOW` - Rate limit time window in seconds (default: 60)

## Model Information

The application uses a machine learning model trained on loan application data. The model:
- Takes 13+ features as input
- Outputs binary classification (approved/rejected)
- Provides probability scores for decision confidence
- Uses feature scaling and categorical encoding for preprocessing

## Security Features

- **Rate Limiting**: Prevents abuse with configurable request limits
- **Input Validation**: Comprehensive validation of all input fields
- **Security Headers**: XSS protection, content type sniffing prevention
- **Error Handling**: Safe error responses without information leakage
- **Logging**: Tracks all predictions and errors for monitoring

## File Structure

```
classification/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── cols_debug.txt         # Debug information for columns
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── X_train_columns.pkl   # Training column information
└── templates/
    └── index.html        # Web interface template
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

<!-- ## Support

For issues or questions:
1. Check the application logs in `predictions.log`
2. Verify all required model files are present
3. Ensure all dependencies are installed
4. Check input validation errors for guidance</content>
<parameter name="filePath">c:\Users\karan\Desktop\Sleepyy\classification\README.md -->