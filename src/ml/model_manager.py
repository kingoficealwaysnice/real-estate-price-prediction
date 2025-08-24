import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        
        # Ensure directories exist
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42),
                'filename': 'random_forest_model.pkl'
            },
            'xgboost': {
                'model': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'filename': 'xgboost_model.pkl'
            },
            'linear_regression': {
                'model': LinearRegression(),
                'filename': 'linear_regression_model.pkl'
            }
        }
        
        # Neural network configuration
        self.nn_config = {
            'layers': [64, 32, 16],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }
    
    def load_models(self):
        """Load all trained models from disk"""
        # Load traditional ML models
        for model_name, config in self.model_configs.items():
            try:
                model_file = os.path.join(self.model_path, config['filename'])
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
                    self.logger.info(f"Loaded {model_name} model")
                else:
                    self.logger.warning(f"Model file not found: {model_file}")
            except Exception as e:
                self.logger.error(f"Error loading {model_name} model: {str(e)}")
        
        # Load neural network model
        try:
            nn_file = os.path.join(self.model_path, 'neural_network_model.h5')
            if os.path.exists(nn_file):
                self.models['neural_network'] = keras.models.load_model(nn_file)
                self.logger.info("Loaded neural network model")
        except Exception as e:
            self.logger.error(f"Error loading neural network model: {str(e)}")
            
        # Check if any models were loaded
        if not self.models:
            self.logger.warning("No models were loaded successfully")
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models with the provided data"""
        try:
            # Train traditional ML models
            for model_name, config in self.model_configs.items():
                self.logger.info(f"Training {model_name}...")
                model = config['model']
                model.fit(X_train, y_train)
                
                # Save the trained model
                model_file = os.path.join(self.model_path, config['filename'])
                joblib.dump(model, model_file)
                self.models[model_name] = model
                
                # Evaluate if validation data is provided
                if X_val is not None and y_val is not None:
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    self.logger.info(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")
            
            # Train neural network
            self._train_neural_network(X_train, y_train, X_val, y_val)
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
    
    def _train_neural_network(self, X_train, y_train, X_val=None, y_val=None):
        """Train neural network model"""
        try:
            model = Sequential()
            
            # Input layer
            model.add(Dense(self.nn_config['layers'][0], activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(self.nn_config['dropout']))
            
            # Hidden layers
            for units in self.nn_config['layers'][1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(self.nn_config['dropout']))
            
            # Output layer
            model.add(Dense(1, activation='linear'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.nn_config['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            model.fit(
                X_train, y_train,
                epochs=self.nn_config['epochs'],
                batch_size=self.nn_config['batch_size'],
                validation_data=validation_data,
                verbose=1
            )
            
            # Save model
            nn_file = os.path.join(self.model_path, 'neural_network_model.h5')
            model.save(nn_file)
            self.models['neural_network'] = model
            
            self.logger.info("Neural network training completed")
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {str(e)}")
    
    def predict(self, features, model_name='ensemble'):
        """Make prediction using specified model or ensemble"""
        try:
            if model_name == 'ensemble':
                # Use ensemble prediction (average of all models)
                predictions = []
                for name, model in self.models.items():
                    if hasattr(model, 'predict'):
                        # Handle neural network differently
                        if name == 'neural_network':
                            pred = model.predict(features.reshape(1, -1))[0][0]
                        else:
                            pred = model.predict([features])[0]
                        predictions.append(pred)
                
                if predictions:
                    return np.mean(predictions)
                else:
                    raise ValueError("No models available for prediction")
            
            elif model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'predict'):
                    # Handle neural network differently
                    if model_name == 'neural_network':
                        return model.predict(features.reshape(1, -1))[0][0]
                    else:
                        return model.predict([features])[0]
                else:
                    raise ValueError(f"Model {model_name} doesn't support prediction")
            
            else:
                raise ValueError(f"Model {model_name} not found")
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test data"""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[model_name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_model_status(self):
        """Get status of all models"""
        status = {}
        
        for model_name in self.model_configs.keys():
            model_file = os.path.join(self.model_path, self.model_configs[model_name]['filename'])
            status[model_name] = {
                'loaded': model_name in self.models,
                'file_exists': os.path.exists(model_file)
            }
        
        # Check neural network
        nn_file = os.path.join(self.model_path, 'neural_network_model.h5')
        status['neural_network'] = {
            'loaded': 'neural_network' in self.models,
            'file_exists': os.path.exists(nn_file)
        }
        
        return status
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from the specified model"""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                return None
        return None