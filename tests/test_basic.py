#!/usr/bin/env python3
"""
Basic unit tests for Real Estate Price Prediction system
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.preprocessing.data_processor import DataProcessor
from src.ml.model_manager import ModelManager

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_processor = DataProcessor()
        
    def test_load_data(self):
        """Test data loading functionality"""
        data = self.data_processor.load_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('price', data.columns)
        
    def test_validate_input_valid(self):
        """Test input validation with valid data"""
        valid_input = {
            'area': 1500,
            'bedrooms': 3,
            'bathrooms': 2,
            'location': 'Downtown'
        }
        is_valid, errors = self.data_processor.validate_input(valid_input)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
    def test_validate_input_invalid(self):
        """Test input validation with invalid data"""
        invalid_input = {
            'area': -100,  # Negative area
            'bedrooms': 3,
            'bathrooms': 2,
            'location': 'InvalidLocation'  # Invalid location
        }
        is_valid, errors = self.data_processor.validate_input(invalid_input)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
    def test_validate_input_missing_fields(self):
        """Test input validation with missing required fields"""
        incomplete_input = {
            'area': 1500,
            'bedrooms': 3
            # Missing bathrooms and location
        }
        is_valid, errors = self.data_processor.validate_input(incomplete_input)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_manager = ModelManager()
        
    def test_model_manager_initialization(self):
        """Test ModelManager initialization"""
        self.assertIsInstance(self.model_manager.models, dict)
        self.assertIsInstance(self.model_manager.model_configs, dict)
        
    def test_get_model_status(self):
        """Test model status retrieval"""
        status = self.model_manager.get_model_status()
        self.assertIsInstance(status, dict)
        self.assertIn('random_forest', status)
        self.assertIn('xgboost', status)
        self.assertIn('linear_regression', status)
        self.assertIn('neural_network', status)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction workflow"""
        # Load and preprocess data
        data = self.data_processor.load_data()
        X, y = self.data_processor.preprocess_data(data)
        
        # Train models
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model_manager.train_models(X_train, y_train, X_test, y_test)
        
        # Test prediction
        test_input = {
            'area': 1500,
            'bedrooms': 3,
            'bathrooms': 2,
            'location': 'Downtown',
            'age': 5,
            'parking_spaces': 1
        }
        
        # Validate input
        is_valid, errors = self.data_processor.validate_input(test_input)
        self.assertTrue(is_valid)
        
        # Preprocess input
        processed_features = self.data_processor.preprocess_input(test_input)
        self.assertIsInstance(processed_features, np.ndarray)
        
        # Make prediction
        prediction = self.model_manager.predict(processed_features)
        self.assertIsInstance(prediction, (int, float))
        self.assertGreater(prediction, 0)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataProcessor))
    test_suite.addTest(unittest.makeSuite(TestModelManager))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 