import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_water_quality(self, file_name='water_quality_data.csv'):
        try:
            # Load the water quality data
            file_path = os.path.join(self.data_dir, file_name)
            data = pd.read_csv(file_path)
            
            # Make a copy to avoid SettingWithCopyWarning
            maharashtra_data = data[data['State'] == 'Maharashtra'].copy()
            
            # Handle missing values - convert to numeric first
            for col in maharashtra_data.columns:
                if col not in ['State', 'District', 'Block']:
                    # Use .loc to avoid SettingWithCopyWarning
                    maharashtra_data.loc[:, col] = pd.to_numeric(maharashtra_data[col], errors='coerce')
            
            # Now fill NA values with mean
            numeric_cols = maharashtra_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                maharashtra_data.loc[:, col] = maharashtra_data[col].fillna(maharashtra_data[col].mean())
            
            # Select features for water quality - ensure they exist in the dataframe
            potential_features = ['pH', 'EC in μS/cm', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K']
            features = [f for f in potential_features if f in maharashtra_data.columns]
            
            if not features:
                raise ValueError("No valid features found in water quality data")
                
            X = maharashtra_data[features]
            
            # Create target variable (TDS > 500 = unsafe)
            if 'TDS' in maharashtra_data.columns:
                y = (maharashtra_data['TDS'] > 500).astype(int)
            else:
                # Default to a dummy target if TDS is missing
                y = np.zeros(len(maharashtra_data))
                print("Warning: TDS column not found, using dummy target")
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            print(f"Water Quality Data: {len(X_train)} training samples, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test, features
            
        except Exception as e:
            print(f"Error in water quality preprocessing: {e}")
            return None, None, None, None, None
    
    def preprocess_geology(self, file_name='Geology Maharashtra.xlsx'):
        try:
            # Load the geology data
            file_path = os.path.join(self.data_dir, file_name)
            data = pd.read_excel(file_path)
            
            # First, identify text columns vs potentially numeric columns
            text_columns = ['Location', 'District', 'Taluka', 'Village']
            text_columns = [col for col in text_columns if col in data.columns]
            
            # Handle each column appropriately
            for col in data.columns:
                if col in text_columns:
                    # Keep text columns as is
                    continue
                else:
                    # For columns that might be lists or have mixed content
                    try:
                        # Try converting to numeric, force non-convertible to NaN
                        data.loc[:, col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        print(f"Warning: Could not process column {col}: {e}")
                        # Keep column as is if conversion fails completely
            
            # Handle missing values only for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                data.loc[:, col] = data[col].fillna(data[col].mean())
            
            # Select features for geology - ensure they exist in the dataframe and are numeric
            available_features = ['LaDeg', 'LaMin', 'LaSec', 'LoDeg', 'LoMin', 'LoSec', 'Elevation', 'Lining', 'MP', 'Dia']
            features = [f for f in available_features if f in numeric_cols]
            
            if not features:
                # If no predefined features found, use all numeric columns except the target
                potential_target = 'Depth'
                features = [col for col in numeric_cols if col != potential_target]
                
            if not features:
                raise ValueError("No valid numeric features found in geology data")
                
            X = data[features]
            
            # Target is the depth if it exists
            if 'Depth' in numeric_cols:
                y = data['Depth']
            else:
                # Create a dummy target based on feature means
                print("Warning: Depth column not found or not numeric, using synthetic target")
                y = X.mean(axis=1)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            print(f"Geology Data: {len(X_train)} training samples, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test, features
            
        except Exception as e:
            print(f"Error in geology preprocessing: {e}")
            return None, None, None, None, None
    
    def preprocess_lithology(self, file_name='Lithology Maharashtra.xlsx'):
        try:
            # Load the lithology data
            file_path = os.path.join(self.data_dir, file_name)
            data = pd.read_excel(file_path)
            
            # If SoilType column doesn't exist and we can't find an alternative, create a dummy column
            if 'SoilType' not in data.columns:
                # Try alternative column names
                soil_columns = [col for col in data.columns if 'soil' in col.lower() or 'type' in col.lower()]
                if soil_columns:
                    # Rename the first matching column to SoilType
                    data.rename(columns={soil_columns[0]: 'SoilType'}, inplace=True)
                else:
                    # Create a dummy SoilType column based on depth ranges or with a default value
                    print("Warning: SoilType column not found. Creating a synthetic SoilType column.")
                    # Check if we have depth information to create soil categories
                    if all(col in data.columns for col in ['From', 'To']):
                        # Create soil types based on depth ranges
                        bins = [0, 10, 20, 50, 100, float('inf')]
                        labels = ['Surface', 'Shallow', 'Medium', 'Deep', 'Very Deep']
                        data['SoilType'] = pd.cut(data['To'], bins=bins, labels=labels)
                    else:
                        # No useful information, just create a default type
                        data['SoilType'] = 'Unknown'
            
            # Handle missing values
            data = data.fillna(0)
            
            # Ensure numeric columns are numeric
            numeric_cols = ['From', 'To', 'Thickness']
            for col in numeric_cols:
                if col in data.columns:
                    data.loc[:, col] = pd.to_numeric(data[col], errors='coerce')
            
            # Select features that exist and are numeric
            potential_features = ['From', 'To', 'Thickness']
            features = [f for f in potential_features if f in data.columns]
            
            if not features:
                # If no predefined features, use numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                features = list(numeric_cols)
                
            if not features:
                # If still no features, create a dummy feature
                print("Warning: No numeric features found. Creating a dummy feature.")
                data['DummyFeature'] = range(len(data))
                features = ['DummyFeature']
                
            X = data[features]
            
            # Encode the soil type
            data['SoilType'] = data['SoilType'].fillna('Unknown')
            y = self.label_encoder.fit_transform(data['SoilType'])
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            print(f"Lithology Data: {len(X_train)} training samples, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test, features
            
        except Exception as e:
            print(f"Error in lithology preprocessing: {e}")
            return None, None, None, None, None
    
    def get_soil_type_mapping(self):
        """Return the mapping between encoded values and soil type names"""
        return {i: soil for i, soil in enumerate(self.label_encoder.classes_)}


if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = DataPreprocessor()
    
    # Water quality
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_water_quality()
    if X_train is not None:
        print(f"Water Quality Features: {features}")
    
    # Geology
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_geology()
    if X_train is not None:
        print(f"Geology Features: {features}")
    
    # Lithology
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_lithology()
    if X_train is not None:
        print(f"Lithology Features: {features}")
        print(f"Soil Type Mapping: {preprocessor.get_soil_type_mapping()}") 