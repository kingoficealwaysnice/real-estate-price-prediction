from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.model_manager import ModelManager
from src.preprocessing.data_processor import DataProcessor
from src.api.routes import api_bp

app = Flask(__name__, 
            template_folder='../web/templates',
            static_folder='../web/static')

# Enable CORS
CORS(app)

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

# Initialize model manager and data processor
model_manager = ModelManager()
data_processor = DataProcessor()

# Load models on startup
model_manager.load_models()

@app.route('/')
def index():
    """Main page of the application"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle price prediction requests"""
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['area', 'bedrooms', 'bathrooms', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess the input data
        processed_data = data_processor.preprocess_input(data)
        
        # Get prediction from model
        prediction = model_manager.predict(processed_data)
        
        return jsonify({
            'predicted_price': float(prediction),
            'confidence': 0.85,  # This would come from model confidence
            'currency': 'USD'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/status')
def model_status():
    """Get status of all models"""
    try:
        status = model_manager.get_model_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data/statistics')
def data_statistics():
    """Get dataset statistics"""
    try:
        stats = data_processor.get_dataset_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models on startup
    model_manager.load_models()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)