import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class AquaSensePredictor:
    def __init__(self, models_dir='../models'):
        self.models_dir = models_dir
        self.water_quality_model = None
        self.geology_model = None
        self.lithology_model = None
        self.scaler = StandardScaler()
        
        # Load all models
        self.load_models()
    
    def load_models(self):
        """Load all the trained models"""
        try:
            # Water quality model
            water_quality_path = os.path.join(self.models_dir, 'water_quality_model.pkl')
            if os.path.exists(water_quality_path):
                self.water_quality_model = joblib.load(water_quality_path)
                print("Water Quality Model loaded successfully")
            else:
                print("Water Quality Model not found")
            
            # Geology model
            geology_path = os.path.join(self.models_dir, 'geology_model.pkl')
            if os.path.exists(geology_path):
                self.geology_model = joblib.load(geology_path)
                print("Geology Model loaded successfully")
            else:
                print("Geology Model not found")
            
            # Lithology model
            lithology_path = os.path.join(self.models_dir, 'lithology_model.pkl')
            if os.path.exists(lithology_path):
                self.lithology_model = joblib.load(lithology_path)
                print("Lithology Model loaded successfully")
            else:
                print("Lithology Model not found")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_water_quality(self, water_params):
        """
        Predict water quality based on input parameters
        
        Args:
            water_params (dict): Dictionary containing water quality parameters
                Required keys: 'pH', 'EC', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K'
        
        Returns:
            dict: Prediction results including safety status and confidence
        """
        if self.water_quality_model is None:
            return {"error": "Water Quality Model not loaded"}
        
        try:
            # Extract features
            features = ['pH', 'EC in μS/cm', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K']
            input_data = np.array([[
                water_params.get('pH', 7.0),
                water_params.get('EC', 500),
                water_params.get('HCO3', 200),
                water_params.get('Cl', 250),
                water_params.get('TH', 300),
                water_params.get('Ca', 75),
                water_params.get('Mg', 30),
                water_params.get('Na', 50),
                water_params.get('K', 10)
            ]])
            
            # Normalize features
            input_scaled = self.scaler.fit_transform(input_data)
            
            # Make prediction
            prediction = self.water_quality_model.predict(input_scaled)[0]
            
            # Get probability if available
            confidence = 0.0
            if hasattr(self.water_quality_model, 'predict_proba'):
                proba = self.water_quality_model.predict_proba(input_scaled)[0]
                confidence = proba[prediction] * 100
            
            # Interpret result
            if prediction == 1:
                status = "Unsafe for Drinking"
                recommendation = "Water treatment recommended before consumption"
            else:
                status = "Safe for Drinking"
                recommendation = "Water meets safety standards for drinking"
            
            return {
                "status": status,
                "confidence": f"{confidence:.2f}%" if confidence > 0 else "N/A",
                "recommendation": recommendation,
                "parameters": {k: water_params.get(k, "N/A") for k in ["pH", "EC", "HCO3", "Cl", "TH", "Ca", "Mg", "Na", "K"]}
            }
            
        except Exception as e:
            return {"error": f"Error in water quality prediction: {e}"}
    
    def predict_geology(self, geo_params):
        """
        Predict geological characteristics based on input parameters
        
        Args:
            geo_params (dict): Dictionary containing geological parameters
                Required keys: Coordinates (LaDeg, LaMin, LaSec, LoDeg, LoMin, LoSec) and well parameters
        
        Returns:
            dict: Prediction results including depth and confidence
        """
        if self.geology_model is None:
            return {"error": "Geology Model not loaded"}
        
        try:
            # Extract features
            features = ['LaDeg', 'LaMin', 'LaSec', 'LoDeg', 'LoMin', 'LoSec', 'Elevation', 'Lining', 'MP', 'Dia']
            input_data = np.array([[
                geo_params.get('LaDeg', 19),
                geo_params.get('LaMin', 30),
                geo_params.get('LaSec', 0),
                geo_params.get('LoDeg', 75),
                geo_params.get('LoMin', 30),
                geo_params.get('LoSec', 0),
                geo_params.get('Elevation', 100),
                geo_params.get('Lining', 5),
                geo_params.get('MP', 1),
                geo_params.get('Dia', 150)
            ]])
            
            # Normalize features
            input_scaled = self.scaler.fit_transform(input_data)
            
            # Make prediction
            depth = self.geology_model.predict(input_scaled)[0]
            
            # Calculate confidence if using XGBoost
            confidence_message = "N/A"
            if hasattr(self.geology_model, 'predict') and hasattr(self.geology_model, 'best_ntree_limit'):
                confidence_message = "High confidence (XGBoost model)"
            
            return {
                "depth": f"{depth:.2f} meters",
                "confidence": confidence_message,
                "recommendation": self._get_well_recommendation(depth),
                "parameters": {k: geo_params.get(k, "N/A") for k in features}
            }
            
        except Exception as e:
            return {"error": f"Error in geology prediction: {e}"}
    
    def predict_lithology(self, litho_params):
        """
        Predict lithological characteristics based on input parameters
        
        Args:
            litho_params (dict): Dictionary containing lithological parameters
                Required keys: 'From', 'To', 'Thickness'
        
        Returns:
            dict: Prediction results including soil type and digging method
        """
        if self.lithology_model is None:
            return {"error": "Lithology Model not loaded"}
        
        try:
            # Extract features
            features = ['From', 'To', 'Thickness']
            input_data = np.array([[
                litho_params.get('From', 0),
                litho_params.get('To', 10),
                litho_params.get('Thickness', 10)
            ]])
            
            # Normalize features
            input_scaled = self.scaler.fit_transform(input_data)
            
            # Make prediction
            soil_type_code = self.lithology_model.predict(input_scaled)[0]
            
            # Get soil type mapping
            soil_type = self._get_soil_type(soil_type_code)
            
            # Get confidence if available
            confidence = 0.0
            if hasattr(self.lithology_model, 'predict_proba'):
                proba = self.lithology_model.predict_proba(input_scaled)[0]
                confidence = proba[soil_type_code] * 100
            
            # Get digging method based on soil type
            digging_method = self._get_digging_method(soil_type)
            
            return {
                "soil_type": soil_type,
                "confidence": f"{confidence:.2f}%" if confidence > 0 else "N/A",
                "digging_method": digging_method,
                "parameters": {k: litho_params.get(k, "N/A") for k in features}
            }
            
        except Exception as e:
            return {"error": f"Error in lithology prediction: {e}"}
    
    def _get_soil_type(self, code):
        """Map soil type code to soil type name"""
        soil_types = {
            0: "Clay And Kankar",
            1: "Fine Sand With Clay",
            2: "Fine Sand",
            3: "Fine To Medium Sand",
            4: "Medium Sand",
            5: "Coarse Sand",
            6: "Coarse To Very Coarse Sand",
            7: "Sandy Clay",
            8: "Clay",
            9: "Gravel"
        }
        
        return soil_types.get(code, "Unknown")
    
    def _get_digging_method(self, soil_type):
        """Recommend digging method based on soil type"""
        easy_digging = ["Clay And Kankar", "Fine Sand With Clay", "Fine Sand"]
        medium_digging = ["Fine To Medium Sand", "Medium Sand", "Sandy Clay", "Clay"]
        hard_digging = ["Coarse Sand", "Coarse To Very Coarse Sand", "Gravel"]
        
        if soil_type in easy_digging:
            return "Manual Digging Possible"
        elif soil_type in medium_digging:
            return "Digging With JCB Recommended"
        elif soil_type in hard_digging:
            return "Digging With Drilling Equipment Required"
        else:
            return "Unknown Method"
    
    def _get_well_recommendation(self, depth):
        """Recommend well type based on depth"""
        if depth < 10:
            return "Shallow well suitable for hand pump"
        elif depth < 50:
            return "Medium depth well suitable for centrifugal pump"
        else:
            return "Deep well requiring submersible pump"


if __name__ == "__main__":
    # Test the predictor
    predictor = AquaSensePredictor()
    
    # Test water quality prediction
    water_params = {
        'pH': 7.5,
        'EC': 550,
        'HCO3': 250,
        'Cl': 200,
        'TH': 350,
        'Ca': 80,
        'Mg': 35,
        'Na': 60,
        'K': 12
    }
    
    water_result = predictor.predict_water_quality(water_params)
    print("\nWater Quality Prediction:")
    for key, value in water_result.items():
        print(f"{key}: {value}")
    
    # Test geology prediction
    geo_params = {
        'LaDeg': 19,
        'LaMin': 32,
        'LaSec': 15,
        'LoDeg': 75,
        'LoMin': 45,
        'LoSec': 30,
        'Elevation': 120,
        'Lining': 8,
        'MP': 2,
        'Dia': 200
    }
    
    geo_result = predictor.predict_geology(geo_params)
    print("\nGeology Prediction:")
    for key, value in geo_result.items():
        print(f"{key}: {value}")
    
    # Test lithology prediction
    litho_params = {
        'From': 5,
        'To': 15,
        'Thickness': 10
    }
    
    litho_result = predictor.predict_lithology(litho_params)
    print("\nLithology Prediction:")
    for key, value in litho_result.items():
        print(f"{key}: {value}") 