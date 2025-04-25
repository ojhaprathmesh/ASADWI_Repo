import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor

class ModelTrainer:
    def __init__(self, models_dir='../models'):
        self.models_dir = models_dir
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def train_water_quality_model(self, X_train, y_train, X_test, y_test):
        print("Training Water Quality Model...")
        
        # Define models to try
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1    # L2 regularization
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # If best model is XGBoost, we can do some hyperparameter tuning
        if best_model_name == 'XGBoost' and best_accuracy < 0.7:
            print("Performing hyperparameter tuning for XGBoost...")
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 5]
            }
            
            grid_search = GridSearchCV(
                estimator=XGBClassifier(objective='binary:logistic', random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            best_accuracy = accuracy_score(y_test, y_pred)
            print(f"Tuned XGBoost Accuracy: {best_accuracy:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'water_quality_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Water Quality Model saved to {model_path}")
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance if using XGBoost
        if hasattr(best_model, 'feature_importances_'):
            self.plot_feature_importance(best_model, 'Water Quality', ['pH', 'EC', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K'])
        
        return best_model, best_accuracy
    
    def train_geology_model(self, X_train, y_train, X_test, y_test):
        print("Training Geology Model...")
        
        # Define models to try
        models = {
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                n_jobs=-1,
                random_state=42,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1    # L2 regularization
            ),
            'Neural Network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        best_model = None
        best_r2 = 0
        best_model_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"{name} RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with R²: {best_r2:.4f}")
        
        # If best model is XGBoost, we can do some hyperparameter tuning
        if best_model_name == 'XGBoost' and best_r2 < 0.7:
            print("Performing hyperparameter tuning for XGBoost...")
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 5]
            }
            
            grid_search = GridSearchCV(
                estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
                param_grid=param_grid,
                cv=3,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            best_r2 = r2_score(y_test, y_pred)
            print(f"Tuned XGBoost RMSE: {rmse:.4f}, R²: {best_r2:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'geology_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Geology Model saved to {model_path}")
        
        # Feature importance if using XGBoost
        if hasattr(best_model, 'feature_importances_'):
            self.plot_feature_importance(best_model, 'Geology', 
                                     ['LaDeg', 'LaMin', 'LaSec', 'LoDeg', 'LoMin', 'LoSec', 
                                      'Elevation', 'Lining', 'MP', 'Dia'])
        
        return best_model, best_r2
    
    def train_lithology_model(self, X_train, y_train, X_test, y_test):
        print("Training Lithology Model...")
        
        # Define models to try
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                n_jobs=-1,
                random_state=42,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1    # L2 regularization
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,  # L2 regularization
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # If best model is XGBoost, we can do some hyperparameter tuning
        if best_model_name == 'XGBoost' and best_accuracy < 0.7:
            print("Performing hyperparameter tuning for XGBoost...")
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 5]
            }
            
            grid_search = GridSearchCV(
                estimator=XGBClassifier(objective='multi:softmax', random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            best_accuracy = accuracy_score(y_test, y_pred)
            print(f"Tuned XGBoost Accuracy: {best_accuracy:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model
        model_path = os.path.join(self.models_dir, 'lithology_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"Lithology Model saved to {model_path}")
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance if using XGBoost
        if hasattr(best_model, 'feature_importances_'):
            self.plot_feature_importance(best_model, 'Lithology', ['From', 'To', 'Thickness'])
        
        return best_model, best_accuracy
    
    def plot_feature_importance(self, model, model_name, feature_names):
        """Plot feature importance for models that support it"""
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(model.feature_importances_)), model.feature_importances_)
        plt.yticks(range(len(model.feature_importances_)), feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title(f'Feature Importance for {model_name} Model')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.models_dir, f'{model_name.lower()}_feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")


if __name__ == "__main__":
    # Initialize preprocessor and model trainer
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Water quality model
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_water_quality()
    if X_train is not None:
        water_quality_model, water_quality_accuracy = trainer.train_water_quality_model(X_train, y_train, X_test, y_test)
        print(f"Water Quality Model Accuracy: {water_quality_accuracy:.4f}")
    
    # Geology model
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_geology()
    if X_train is not None:
        geology_model, geology_r2 = trainer.train_geology_model(X_train, y_train, X_test, y_test)
        print(f"Geology Model R²: {geology_r2:.4f}")
    
    # Lithology model
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_lithology()
    if X_train is not None:
        lithology_model, lithology_accuracy = trainer.train_lithology_model(X_train, y_train, X_test, y_test)
        print(f"Lithology Model Accuracy: {lithology_accuracy:.4f}") 