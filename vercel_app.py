import os
from flask import Flask, render_template, request, jsonify

# Create Flask app with absolute paths
template_dir = os.path.join(os.path.dirname(__file__), 'web/templates')
static_dir = os.path.join(os.path.dirname(__file__), 'web/static')

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# Simple demo prediction
class DemoModelManager:
    def __init__(self):
        self.models_loaded = False
        
    def predict(self, features):
        # Simple price calculation for demo
        area = float(features[0]) if len(features) > 0 else 1000
        bedrooms = float(features[1]) if len(features) > 1 else 3
        bathrooms = float(features[2]) if len(features) > 2 else 2
        
        base_price = area * 150  # $150 per sq ft base
        bedroom_bonus = bedrooms * 10000
        bathroom_bonus = bathrooms * 5000
        
        price = base_price + bedroom_bonus + bathroom_bonus
        return max(price, 50000)  # Minimum $50k

class DemoDataProcessor:
    def preprocess_input(self, data):
        # Process input data
        area = float(data.get('area', 1000))
        bedrooms = float(data.get('bedrooms', 3))
        bathrooms = float(data.get('bathrooms', 2))
        location = data.get('location', 'suburban')
        
        # Simple location factor
        location_factor = 1.2 if location.lower() in ['downtown', 'city_center'] else 1.0
        
        return [area * location_factor, bedrooms, bathrooms]

# Initialize demo managers
model_manager = DemoModelManager()
data_processor = DemoDataProcessor()

@app.route('/')
def index():
    """Main page of the application"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'demo_mode': True,
        'message': 'Real Estate Price Prediction API is running'
    })

@app.route('/api/predict', methods=['POST'])
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
            'confidence': 0.5,
            'currency': 'USD',
            'demo_mode': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/importance')
def feature_importance():
    """Get feature importance"""
    return jsonify({
        'feature_importance': {
            'area': 0.45,
            'bedrooms': 0.25,
            'bathrooms': 0.15,
            'location': 0.15
        },
        'success': True
    })

# Vercel serverless handler
app = app