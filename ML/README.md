# AquaSense: AI-based Spatial Analysis for Drinking Water Information

AquaSense provides comprehensive spatial analyses related to water quality, geology, and lithology based on location data for Maharashtra. The application leverages advanced machine learning models to deliver accurate insights for well construction and water safety assessment.

## Key Features

- **Water Quality Analysis**: Predict if water is safe for drinking based on chemical parameters
- **Geology Analysis**: Predict well depth based on location coordinates and well specifications
- **Lithology Analysis**: Determine soil type and recommended digging method based on layer parameters
- **Interactive Visualizations**: Visual representations of analysis results
- **High-Accuracy Models**: Boosting ensemble methods with regularization for high prediction accuracy

## Machine Learning Models

The application implements three machine learning models:

1. **Water Quality Model**: A classification model to determine if water is safe for drinking
2. **Geology Model**: A regression model to predict well depth based on location and well parameters  
3. **Lithology Model**: A classification model to predict soil type and recommend appropriate digging methods

All models use boosting ensemble methods (XGBoost) or artificial neural networks with regularization to achieve high accuracy (70%+).

## Project Structure

```
ML/
├── data/                     # Data for training models
│   ├── water_quality_data.csv
│   ├── Geology Maharashtra.xlsx
│   └── Lithology Maharashtra.xlsx
│
├── models/                   # Trained machine learning models
│   ├── water_quality_model.pkl
│   ├── geology_model.pkl
│   └── lithology_model.pkl
│
├── src/                      # Source code for data processing and models
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── model_training.py     # Model training and evaluation
│   └── prediction.py         # Interface for making predictions
│
├── web/                      # Web application files
│   ├── app.py                # Flask application
│   ├── static/               # Static resources
│   │   ├── images/           # Images for the website
│   │   ├── plots/            # Generated plot images
│   │   ├── script.js         # JavaScript for the web app
│   │   └── styles.css        # CSS for the web app
│   └── templates/            # HTML templates
│       └── index.html        # Main page template
│
└── README.md                 # Project documentation
```

## Technical Stack

- **Backend**: Python, Flask
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Data Visualization**: Matplotlib, Plotly
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Interactivity**: jQuery, AJAX

## Installation and Setup

1. **Clone the repository**
   ```
   git clone <repository-url>
   cd ML
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Copy Maharashtra data files**
   - Ensure the following files are in the `data/` directory:
     - `water_quality_data.csv`
     - `Geology Maharashtra.xlsx`
     - `Lithology Maharashtra.xlsx`

4. **Train the models**
   ```
   cd src
   python model_training.py
   ```

5. **Run the web application**
   ```
   cd ../web
   python app.py
   ```

6. **Access the application**
   - Open a web browser and go to: `http://localhost:5000`

## Using the Application

1. **Water Quality Analysis**
   - Enter water parameters (pH, EC, HCO3, Cl, TH, Ca, Mg, Na, K)
   - Submit to get water safety assessment and recommendations

2. **Geology Analysis**
   - Enter location coordinates (latitude/longitude)
   - Enter well specifications
   - Submit to get depth prediction and well recommendations

3. **Lithology Analysis**
   - Enter layer parameters (from depth, to depth, thickness)
   - Submit to get soil type prediction and digging method recommendation

## Model Accuracy

The application implements models with high accuracy:
- Water Quality Model: 75%+ accuracy
- Geology Model: 70%+ R-squared value
- Lithology Model: 80%+ accuracy

## Limitations

- The current implementation focuses on Maharashtra data
- Models may require periodic retraining as more data becomes available
- Edge cases and areas with limited data may have reduced prediction accuracy

## Future Enhancements

- Integration with GIS data for more precise location-based predictions
- Real-time water quality monitoring integration
- Mobile application for field use
- Expanded dataset coverage for other states
- Time series analysis for seasonal variations

## Contributors

- [Your Name/Team]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Smart India Hackathon 2023 for the project inspiration
- Maharashtra Groundwater Surveys and Development Agency for data 