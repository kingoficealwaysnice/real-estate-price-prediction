#!/usr/bin/env python3
"""
Real Estate Price Prediction - Model Training Script
This script trains all machine learning models and saves them for the web application.
"""

import os
import sys
import logging
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ml.model_manager import ModelManager
from src.preprocessing.data_processor import DataProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main training function"""
    logger = setup_logging()
    
    logger.info("Starting Real Estate Price Prediction Model Training")
    logger.info("=" * 60)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_processor = DataProcessor()
        model_manager = ModelManager()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data = data_processor.load_data()
        logger.info(f"Loaded dataset with {len(data)} records")
        
        # Display dataset info
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Features: {list(data.columns)}")
        logger.info(f"Price range: ${data['price'].min():,.0f} - ${data['price'].max():,.0f}")
        logger.info(f"Average price: ${data['price'].mean():,.0f}")
        
        # Preprocess data
        X, y = data_processor.preprocess_data(data)
        logger.info(f"Preprocessed features shape: {X.shape}")
        
        # Split data for training and validation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train models
        logger.info("Training machine learning models...")
        start_time = datetime.now()
        
        model_manager.train_models(X_train, y_train, X_test, y_test)
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = model_manager.evaluate_models(X_test, y_test)
        
        # Display evaluation results
        logger.info("Model Evaluation Results:")
        logger.info("-" * 40)
        
        for model_name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                logger.info(f"{model_name.upper()}:")
                logger.info(f"  R² Score: {metrics['r2']:.4f}")
                logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                logger.info(f"  MAE: {metrics['mae']:.2f}")
                logger.info(f"  MSE: {metrics['mse']:.2f}")
            else:
                logger.error(f"{model_name}: {metrics['error']}")
        
        # Get feature importance
        logger.info("Feature Importance (Random Forest):")
        feature_importance = model_manager.get_feature_importance('random_forest')
        if feature_importance is not None:
            # Get feature names
            feature_names = data.drop(columns=['price']).columns.tolist()
            
            # Create feature importance list
            importance_list = []
            for i, importance in enumerate(feature_importance):
                if i < len(feature_names):
                    importance_list.append((feature_names[i], importance))
            
            # Sort by importance
            importance_list.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in importance_list[:10]:  # Top 10 features
                logger.info(f"  {feature}: {importance:.4f}")
        
        # Save preprocessing pipeline
        if data_processor.preprocessing_pipeline is not None:
            data_processor._save_preprocessing_pipeline()
            logger.info("Preprocessing pipeline saved")
        
        # Test prediction
        logger.info("Testing prediction with sample data...")
        sample_data = {
            'area': 1500,
            'bedrooms': 3,
            'bathrooms': 2,
            'location': 'Downtown',
            'age': 5,
            'parking_spaces': 1
        }
        
        try:
            processed_features = data_processor.preprocess_input(sample_data)
            prediction = model_manager.predict(processed_features)
            logger.info(f"Sample prediction: ${prediction:,.0f}")
        except Exception as e:
            logger.error(f"Prediction test failed: {str(e)}")
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("Models are ready for the web application.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 