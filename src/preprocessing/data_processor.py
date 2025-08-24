import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any

class DataProcessor:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        self.models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Ensure directories exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.imputer = SimpleImputer(strategy='median')
        
        # Feature columns configuration
        self.numeric_features = ['area', 'bedrooms', 'bathrooms', 'age', 'parking_spaces']
        self.categorical_features = ['location', 'property_type', 'furnishing_status']
        self.target_column = 'price'
        
        # Load preprocessing pipeline if exists
        self.preprocessing_pipeline = None
        self._load_preprocessing_pipeline()
    
    def _load_preprocessing_pipeline(self):
        """Load the preprocessing pipeline from disk"""
        pipeline_file = os.path.join(self.models_path, 'preprocessing_pipeline.pkl')
        if os.path.exists(pipeline_file):
            try:
                self.preprocessing_pipeline = joblib.load(pipeline_file)
                self.logger.info("Loaded preprocessing pipeline")
            except Exception as e:
                self.logger.error(f"Error loading preprocessing pipeline: {str(e)}")
    
    def _save_preprocessing_pipeline(self):
        """Save the preprocessing pipeline to disk"""
        if self.preprocessing_pipeline is not None:
            pipeline_file = os.path.join(self.models_path, 'preprocessing_pipeline.pkl')
            try:
                joblib.dump(self.preprocessing_pipeline, pipeline_file)
                self.logger.info("Saved preprocessing pipeline")
            except Exception as e:
                self.logger.error(f"Error saving preprocessing pipeline: {str(e)}")
    
    def load_data(self, filename: str = 'real_estate_data.csv') -> pd.DataFrame:
        """Load dataset from file"""
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            # Create sample data if file doesn't exist
            self.logger.info("Creating sample dataset...")
            return self._create_sample_data()
        
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample real estate dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data
        data = {
            'area': np.random.normal(1500, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'parking_spaces': np.random.randint(0, 3, n_samples),
            'location': np.random.choice(['Downtown', 'Suburb', 'Rural', 'City Center', 'Residential'], n_samples),
            'property_type': np.random.choice(['Apartment', 'House', 'Condo', 'Townhouse'], n_samples),
            'furnishing_status': np.random.choice(['Furnished', 'Semi-furnished', 'Unfurnished'], n_samples)
        }
        
        # Create price based on features with some noise
        base_price = (
            data['area'] * 100 +
            data['bedrooms'] * 50000 +
            data['bathrooms'] * 30000 +
            (50 - data['age']) * 1000 +
            data['parking_spaces'] * 15000
        )
        
        # Add location premium
        location_premium = {
            'Downtown': 1.5,
            'City Center': 1.4,
            'Suburb': 1.2,
            'Residential': 1.1,
            'Rural': 0.8
        }
        
        for i, loc in enumerate(data['location']):
            base_price[i] *= location_premium[loc]
        
        # Add noise
        noise = np.random.normal(0, 0.1, n_samples)
        data['price'] = base_price * (1 + noise)
        
        # Ensure positive prices
        data['price'] = np.maximum(data['price'], 50000)
        
        df = pd.DataFrame(data)
        
        # Save sample data
        file_path = os.path.join(self.data_path, 'real_estate_data.csv')
        df.to_csv(file_path, index=False)
        self.logger.info(f"Created and saved sample dataset with {len(df)} rows")
        
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the dataset and return features and target"""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Feature engineering
            data = self._engineer_features(data)
            
            # Separate features and target
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            # Create preprocessing pipeline
            self._create_preprocessing_pipeline(X)
            
            # Transform features
            X_transformed = self.preprocessing_pipeline.fit_transform(X)
            
            self.logger.info(f"Preprocessed data: {X_transformed.shape}")
            return X_transformed, y.values
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing numeric values with median
        for col in self.numeric_features:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        
        # Fill missing categorical values with mode
        for col in self.categorical_features:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing ones"""
        # Price per square foot
        if 'area' in data.columns and 'price' in data.columns:
            data['price_per_sqft'] = data['price'] / data['area']
        
        # Total rooms
        if 'bedrooms' in data.columns and 'bathrooms' in data.columns:
            data['total_rooms'] = data['bedrooms'] + data['bathrooms']
        
        # Age category
        if 'age' in data.columns:
            data['age_category'] = pd.cut(data['age'], 
                                        bins=[0, 5, 15, 30, 100], 
                                        labels=['New', 'Recent', 'Old', 'Very Old'])
        
        # Area category
        if 'area' in data.columns:
            data['area_category'] = pd.cut(data['area'], 
                                         bins=[0, 1000, 1500, 2000, 5000], 
                                         labels=['Small', 'Medium', 'Large', 'Very Large'])
        
        return data
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame):
        """Create preprocessing pipeline for the features"""
        # Define numeric and categorical transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Create column transformer
        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess single input for prediction"""
        try:
            # Add default values for missing optional fields
            default_values = {
                'age': 0,
                'parking_spaces': 0,
                'property_type': 'House',
                'furnishing_status': 'Semi-furnished'
            }
            
            # Update input_data with defaults for missing fields
            for field, default_value in default_values.items():
                if field not in input_data or input_data[field] is None:
                    input_data[field] = default_value
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Handle missing values
            input_df = self._handle_missing_values(input_df)
            
            # Engineer features
            input_df = self._engineer_features(input_df)
            
            # Transform using pipeline
            if self.preprocessing_pipeline is not None:
                features = self.preprocessing_pipeline.transform(input_df)
                return features.flatten()
            else:
                raise ValueError("Preprocessing pipeline not available")
                
        except Exception as e:
            self.logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset"""
        try:
            data = self.load_data()
            
            stats = {
                'total_records': len(data),
                'total_features': len(data.columns) - 1,  # Exclude target
                'numeric_features': len(self.numeric_features),
                'categorical_features': len(self.categorical_features),
                'missing_values': data.isnull().sum().to_dict(),
                'price_statistics': {
                    'mean': float(data['price'].mean()),
                    'median': float(data['price'].median()),
                    'std': float(data['price'].std()),
                    'min': float(data['price'].min()),
                    'max': float(data['price'].max())
                },
                'feature_statistics': {}
            }
            
            # Add statistics for each feature
            for col in data.columns:
                if col != 'price':
                    if data[col].dtype in ['int64', 'float64']:
                        stats['feature_statistics'][col] = {
                            'mean': float(data[col].mean()),
                            'median': float(data[col].median()),
                            'std': float(data[col].std())
                        }
                    else:
                        stats['feature_statistics'][col] = {
                            'unique_values': int(data[col].nunique()),
                            'most_common': data[col].mode()[0] if len(data[col].mode()) > 0 else None
                        }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting dataset statistics: {str(e)}")
            return {'error': str(e)}
    
    def validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data for prediction"""
        errors = []
        
        # Check required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'location']
        for field in required_fields:
            if field not in input_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric fields
        numeric_fields = ['area', 'bedrooms', 'bathrooms']
        for field in numeric_fields:
            if field in input_data:
                try:
                    value = float(input_data[field])
                    if value < 0:
                        errors.append(f"{field} must be positive")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Validate categorical fields
        if 'location' in input_data:
            valid_locations = ['Downtown', 'Suburb', 'Rural', 'City Center', 'Residential']
            if input_data['location'] not in valid_locations:
                errors.append(f"Invalid location. Must be one of: {valid_locations}")
        
        # Validate optional categorical fields if provided
        if 'property_type' in input_data and input_data['property_type']:
            valid_property_types = ['Apartment', 'House', 'Condo', 'Townhouse']
            if input_data['property_type'] not in valid_property_types:
                errors.append(f"Invalid property type. Must be one of: {valid_property_types}")
        
        if 'furnishing_status' in input_data and input_data['furnishing_status']:
            valid_furnishing_statuses = ['Furnished', 'Semi-furnished', 'Unfurnished']
            if input_data['furnishing_status'] not in valid_furnishing_statuses:
                errors.append(f"Invalid furnishing status. Must be one of: {valid_furnishing_statuses}")
        
        return len(errors) == 0, errors 