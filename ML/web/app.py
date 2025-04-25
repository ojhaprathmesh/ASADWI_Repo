from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction import AquaSensePredictor

app = Flask(__name__)

# Initialize predictor
predictor = AquaSensePredictor(models_dir='../models')

# Create a plots directory
os.makedirs('static/plots', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Function to map frontend parameters to backend parameters
def map_water_parameters(form_data):
    """Maps frontend water quality parameters to backend parameters"""
    mapping = {
        # Frontend form field : Backend parameter
        'ph': 'pH',                # pH stays the same
        'hardness': 'TH',          # Hardness maps to Total Hardness (TH)
        'solids': 'EC',            # Total Dissolved Solids maps to Electrical Conductivity (EC)
        'chloramines': 'Cl',       # Chloramines maps to Chloride (Cl)
        'sulfate': 'HCO3',         # Sulfate maps to Bicarbonate (HCO3)
        'conductivity': 'EC',      # Conductivity is directly EC
        'organic_carbon': 'K',     # Organic Carbon maps to Potassium (K)
        'trihalomethanes': 'Na',   # Trihalomethanes maps to Sodium (Na)
        'turbidity': 'Mg'          # Turbidity maps to Magnesium (Mg)
    }
    
    # Create a backend parameter dictionary with default values
    backend_params = {
        'pH': 7.0,
        'EC': 500,
        'HCO3': 200,
        'Cl': 250,
        'TH': 300,
        'Ca': 75,
        'Mg': 30,
        'Na': 50,
        'K': 10
    }
    
    # Update with values from the form data
    for frontend_param, backend_param in mapping.items():
        if frontend_param in form_data and form_data[frontend_param]:
            try:
                value = float(form_data[frontend_param])
                backend_params[backend_param] = value
                
                # Special case for conductivity/solids to set EC only once
                if frontend_param == 'conductivity' and 'solids' in form_data and form_data['solids']:
                    # Prioritize the conductivity value over solids for EC
                    pass
                elif frontend_param == 'solids' and 'conductivity' not in form_data:
                    # Only use solids for EC if conductivity not provided
                    backend_params['EC'] = value
            except (ValueError, TypeError):
                # If conversion fails, keep default value
                pass
    
    # Calculate Ca if it's not directly provided
    if 'Ca' not in form_data:
        # Estimate Ca from TH (assuming Ca is about 70% of total hardness)
        backend_params['Ca'] = backend_params['TH'] * 0.7
    
    return backend_params

@app.route('/water_quality', methods=['POST'])
def water_quality():
    try:
        # Get the form data and map it to backend parameters
        form_data = request.form.to_dict()
        water_params = map_water_parameters(form_data)
        
        # Get prediction
        result = predictor.predict_water_quality(water_params)
        
        # Generate visualization
        chart = generate_water_quality_chart(water_params, result['status'])
        
        # Add chart to result
        result['chart'] = chart
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/geology', methods=['POST'])
def geology():
    # Get the form data
    geo_params = {
        'LaDeg': float(request.form.get('LaDeg', 19)),
        'LaMin': float(request.form.get('LaMin', 30)),
        'LaSec': float(request.form.get('LaSec', 0)),
        'LoDeg': float(request.form.get('LoDeg', 75)),
        'LoMin': float(request.form.get('LoMin', 30)),
        'LoSec': float(request.form.get('LoSec', 0)),
        'Elevation': float(request.form.get('Elevation', 100)),
        'Lining': float(request.form.get('Lining', 5)),
        'MP': float(request.form.get('MP', 1)),
        'Dia': float(request.form.get('Dia', 150))
    }
    
    # Get prediction
    result = predictor.predict_geology(geo_params)
    
    # Generate visualization
    chart = generate_geology_chart(geo_params, result['depth'])
    
    # Add chart to result
    result['chart'] = chart
    
    return jsonify(result)

@app.route('/lithology', methods=['POST'])
def lithology():
    # Get the form data
    litho_params = {
        'From': float(request.form.get('From', 0)),
        'To': float(request.form.get('To', 10)),
        'Thickness': float(request.form.get('Thickness', 10))
    }
    
    # Get prediction
    result = predictor.predict_lithology(litho_params)
    
    # Generate visualization
    chart = generate_lithology_chart(litho_params, result['soil_type'])
    
    # Add chart to result
    result['chart'] = chart
    
    return jsonify(result)

# Add new API endpoints to handle form submissions from index.html
@app.route('/predict/water_quality', methods=['POST'])
def predict_water_quality():
    try:
        # Convert form data to dict for mapping
        form_data = request.form.to_dict()
        
        # Map frontend parameters to backend parameters
        water_params = map_water_parameters(form_data)
        
        # Get prediction
        result = predictor.predict_water_quality(water_params)
        
        # Generate visualization
        chart = generate_water_quality_chart(water_params, result['status'])
        
        # Add chart URL to result
        result['chart_url'] = chart
        
        # Format the response for the frontend
        response = {
            'result': result['status'],
            'details': {
                'Status': result['status'],
                'Confidence': result['confidence'],
                'Recommendation': result['recommendation']
            },
            'chart': chart
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error in water quality prediction: {e}")
        return jsonify({"error": str(e)}), 500

def generate_water_quality_chart(water_params, status):
    """Generate a radar chart for water quality parameters"""
    # Parameters to include in the chart
    params = ['pH', 'EC', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K']
    
    # Reference values for safe drinking water (normalized to 0-1 scale)
    reference = {
        'pH': 7.5,
        'EC': 500,
        'HCO3': 200,
        'Cl': 250,
        'TH': 300,
        'Ca': 75,
        'Mg': 30,
        'Na': 50,
        'K': 10
    }
    
    # Normalize values (0-1 scale)
    normalized_values = []
    for param in params:
        if param == 'pH':
            # Special handling for pH (optimal is ~7.5)
            normalized_values.append(1 - abs(water_params[param] - reference[param]) / 7)
        else:
            # For other parameters, normalize based on reference
            normalized_values.append(min(1, water_params[param] / (reference[param] * 1.5)))
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(params)
    
    # Compute angle for each parameter
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add values for the chart
    values = normalized_values + [normalized_values[0]]  # Close the loop
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='Your Water')
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params)
    
    # Add title
    plt.title(f'Water Quality Analysis\nStatus: {status}', size=15, color='blue', y=1.1)
    
    # Save chart to image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    img_path = f'static/plots/water_quality_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    return img_path

def generate_geology_chart(geo_params, depth):
    """Generate a visualization for geology prediction"""
    # Extract depth value from string
    depth_value = float(depth.split()[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup diagram dimensions
    max_depth = 100
    normalized_depth = min(depth_value, max_depth)
    
    # Draw ground level
    ax.axhline(y=0, color='brown', linewidth=2)
    
    # Draw well shaft
    ax.plot([0.4, 0.6], [0, 0], color='black', linewidth=3)
    ax.plot([0.4, 0.4], [0, -normalized_depth], color='black', linewidth=2)
    ax.plot([0.6, 0.6], [0, -normalized_depth], color='black', linewidth=2)
    ax.plot([0.4, 0.6], [-normalized_depth, -normalized_depth], color='blue', linewidth=3)
    
    # Fill water in well
    ax.fill_between([0.4, 0.6], [-normalized_depth, -normalized_depth], [-normalized_depth*0.8, -normalized_depth*0.8], color='blue', alpha=0.5)
    
    # Draw ground layers
    colors = ['sienna', 'peru', 'burlywood', 'tan', 'moccasin']
    layer_depths = [0, -20, -40, -60, -80, -100]
    for i in range(len(layer_depths)-1):
        ax.fill_between([0, 1], [layer_depths[i], layer_depths[i]], 
                      [layer_depths[i+1], layer_depths[i+1]], 
                      color=colors[i], alpha=0.5)
    
    # Add depth marker
    ax.annotate(f'Depth: {depth}', xy=(0.7, -normalized_depth), 
               xytext=(0.75, -normalized_depth),
               arrowprops=dict(facecolor='black', shrink=0.05),
               fontsize=12, ha='left')
    
    # Add location info
    lat = f"{geo_params['LaDeg']}°{geo_params['LaMin']}'{geo_params['LaSec']}\""
    lon = f"{geo_params['LoDeg']}°{geo_params['LoMin']}'{geo_params['LoSec']}\""
    location_text = f"Location: {lat} N, {lon} E\nElevation: {geo_params['Elevation']} m"
    ax.text(0.05, -10, location_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_title('Well Depth Prediction', fontsize=16)
    ax.set_xlabel('Cross Section', fontsize=12)
    ax.set_ylabel('Depth (meters)', fontsize=12)
    
    # Adjust y-axis limits
    ax.set_ylim(-max_depth-10, 10)
    
    # Remove x-axis ticks
    ax.set_xticks([])
    
    # Save chart to image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    img_path = f'static/plots/geology_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    return img_path

def generate_lithology_chart(litho_params, soil_type):
    """Generate a visualization for lithology prediction"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Setup soil layer dimensions
    from_depth = litho_params['From']
    to_depth = litho_params['To']
    thickness = litho_params['Thickness']
    
    # Soil type colors
    soil_colors = {
        "Clay And Kankar": "sienna",
        "Fine Sand With Clay": "peru",
        "Fine Sand": "sandybrown",
        "Fine To Medium Sand": "burlywood",
        "Medium Sand": "tan",
        "Coarse Sand": "moccasin",
        "Coarse To Very Coarse Sand": "khaki",
        "Sandy Clay": "darkgoldenrod",
        "Clay": "saddlebrown",
        "Gravel": "gray"
    }
    
    # Get color for the predicted soil type
    soil_color = soil_colors.get(soil_type, "lightgray")
    
    # Draw soil layers
    ax.fill_between([0, 1], [from_depth, from_depth], [to_depth, to_depth], color=soil_color, alpha=0.7)
    
    # Add soil type annotation
    ax.text(0.5, (from_depth + to_depth) / 2, soil_type, fontsize=14, 
           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add depth markers
    ax.axhline(y=from_depth, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=to_depth, color='black', linestyle='--', linewidth=1)
    
    # Add depth labels
    ax.text(1.05, from_depth, f"{from_depth} m", fontsize=10, va='center')
    ax.text(1.05, to_depth, f"{to_depth} m", fontsize=10, va='center')
    
    # Set labels and title
    ax.set_title(f'Soil Type Prediction: {soil_type}', fontsize=16)
    ax.set_xlabel('Cross Section', fontsize=12)
    ax.set_ylabel('Depth (meters)', fontsize=12)
    
    # Adjust y-axis (reverse to show depth increasing downward)
    ax.set_ylim(to_depth + 5, from_depth - 5)
    
    # Remove x-axis ticks
    ax.set_xticks([])
    
    # Add legend with soil types
    patches = [plt.Rectangle((0,0), 1, 1, color=color, alpha=0.7) for color in soil_colors.values()]
    plt.legend(patches, soil_colors.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
    
    # Save chart to image
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    img_path = f'static/plots/lithology_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    
    return img_path

if __name__ == '__main__':
    app.run(debug=True, port=5000) 