from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.model_manager import ModelManager
from src.preprocessing.data_processor import DataProcessor

api_bp = Blueprint('api', __name__)

# Initialize components
model_manager = ModelManager()
data_processor = DataProcessor()

# Load models when the module is imported
model_manager.load_models()

@api_bp.route('/predict', methods=['POST'])
@cross_origin()
def predict_price():
    """Predict real estate price based on input features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, errors = data_processor.validate_input(data)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Preprocess input
        processed_features = data_processor.preprocess_input(data)
        
        # Make prediction
        prediction = model_manager.predict(processed_features)
        
        return jsonify({
            'success': True,
            'predicted_price': float(prediction),
            'currency': 'USD',
            'confidence': 0.85,
            'input_features': data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict/batch', methods=['POST'])
@cross_origin()
def predict_batch():
    """Predict prices for multiple properties"""
    try:
        data = request.get_json()
        
        if not data or 'properties' not in data:
            return jsonify({'error': 'No properties data provided'}), 400
        
        properties = data['properties']
        if not isinstance(properties, list):
            return jsonify({'error': 'Properties must be a list'}), 400
        
        predictions = []
        errors = []
        
        for i, property_data in enumerate(properties):
            try:
                # Validate input
                is_valid, validation_errors = data_processor.validate_input(property_data)
                if not is_valid:
                    errors.append({
                        'index': i,
                        'errors': validation_errors
                    })
                    continue
                
                # Preprocess input
                processed_features = data_processor.preprocess_input(property_data)
                
                # Make prediction
                prediction = model_manager.predict(processed_features)
                
                predictions.append({
                    'index': i,
                    'predicted_price': float(prediction),
                    'input_features': property_data
                })
                
            except Exception as e:
                errors.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'errors': errors,
            'total_processed': len(predictions),
            'total_errors': len(errors)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models', methods=['GET'])
@cross_origin()
def get_models():
    """Get information about available models"""
    try:
        status = model_manager.get_model_status()
        
        model_info = {}
        for model_name, model_status in status.items():
            model_info[model_name] = {
                'loaded': model_status['loaded'],
                'available': model_status['file_exists'],
                'type': 'ensemble' if model_name == 'ensemble' else 'individual'
            }
        
        return jsonify({
            'success': True,
            'models': model_info,
            'total_models': len(model_info)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/train', methods=['POST'])
@cross_origin()
def train_models():
    """Train all models with current dataset"""
    try:
        # Load and preprocess data
        data = data_processor.load_data()
        X, y = data_processor.preprocess_data(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        model_manager.train_models(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        evaluation_results = model_manager.evaluate_models(X_test, y_test)
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully',
            'evaluation_results': evaluation_results,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/models/evaluate', methods=['GET'])
@cross_origin()
def evaluate_models():
    """Evaluate all models on test data"""
    try:
        # Load and preprocess data
        data = data_processor.load_data()
        X, y = data_processor.preprocess_data(data)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Evaluate models
        evaluation_results = model_manager.evaluate_models(X_test, y_test)
        
        return jsonify({
            'success': True,
            'evaluation_results': evaluation_results,
            'test_samples': len(X_test)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/data/statistics', methods=['GET'])
@cross_origin()
def get_data_statistics():
    """Get dataset statistics"""
    try:
        stats = data_processor.get_dataset_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/data/sample', methods=['GET'])
@cross_origin()
def get_sample_data():
    """Get sample data for testing"""
    try:
        data = data_processor.load_data()
        sample = data.head(10).to_dict('records')
        
        return jsonify({
            'success': True,
            'sample_data': sample,
            'total_records': len(data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/features/importance', methods=['GET'])
@cross_origin()
def get_feature_importance():
    """Get feature importance from Random Forest model"""
    try:
        feature_importance = model_manager.get_feature_importance('random_forest')
        
        if feature_importance is not None:
            # Get feature names
            data = data_processor.load_data()
            feature_names = data.drop(columns=['price']).columns.tolist()
            
            # Create feature importance dictionary
            importance_dict = {}
            for i, importance in enumerate(feature_importance):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)
            
            return jsonify({
                'success': True,
                'feature_importance': importance_dict
            })
        else:
            return jsonify({
                'error': 'Feature importance not available. Train the Random Forest model first.'
            }), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        model_status = model_manager.get_model_status()
        models_loaded = any(status['loaded'] for status in model_status.values())
        
        # Check if data processor is working
        try:
            data_processor.load_data()
            data_available = True
        except:
            data_available = False
        
        return jsonify({
            'status': 'healthy',
            'models_loaded': models_loaded,
            'data_available': data_available,
            'timestamp': str(pd.Timestamp.now())
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500