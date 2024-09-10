# _ASADWI: AI-based Spatial Analysis for Drinking Water Information_

## Project Name Origin:
**_ASADWI_** is derived from the Hindi word "आसाढ़", representing a rain-dominant month. This project, ASADWI (AI-based Spatial Analysis for Drinking Water Information), is developed for the Smart India Hackathon 2023. The application focuses on providing spatial analyses related to water quality, geology, and lithology based on user-provided location information.
___Original Name: AquaSense - AI Driven Well Insights___

## Project Structure:
### Main Directory:
- **ai_models:**
  - Three Python Scripts: Train and store machine learning models for water quality, geology, and lithology.
  - Three Pickle Files: Storage for the trained ML models.
- **data:**
  - CSV and XLSX Files: Datasets for training machine learning models.
- **raw_data:**
  - PDF Files: Original data extracted from PDFs.
- **static:**
  - logo-asadwi.png: Project logo.
  - script_new.js: JavaScript for web interactions.
  - style_new.css: CSS styling for the web application.
  - **templates:**
    - main_new.html: HTML template for the Flask web application.
#### Files (in Main Directory):
- Flask App.py: Main Flask application.
- Interface.py: Python scripts for machine learning model interface.

## Machine Learning Models:
Three machine learning models predict water quality, geology information, and lithology based on user-provided location data. While these models are functional, they are acknowledged as inefficient due to limited datasets, resulting in suboptimal time complexity.

## How to Use
1. **Clone the Repository:**
   ```
   git clone https://github.com/ojhaprathmesh/AI-based_Spatial_Analysis_for_Drinking_Water_Information.git
   ```
   
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
   
3. **Run the Application (Make sure you run the terminal in main directory):**
   ```
   python Flask_App.py
   ```

4. **Access the Web Application:**
   Open a web browser and navigate to `http://localhost:5000` to access the ASADWI web interface.

5. **Input Location Data:**
   Enter the required location details in the provided input field.

6. **View Analysis Results:**
   Upon submission, the application will display spatial analysis results for water quality, geology, and lithology based on the provided location data.

7. **Contribute:**
   Fork the repository, make desired improvements or modifications, and submit a pull request. Contributions towards improving machine learning models, enhancing UI/UX, and optimizing algorithms are highly appreciated.

8. **Feedback:**
   We welcome feedback and suggestions for further enhancements. Feel free to reach out with any ideas or contributions to make ASADWI more efficient and user-friendly.

## Call for Contributions:
- **Improved Machine Learning Models:**
  - Contributors are invited to enhance the existing models through:
    - Advanced algorithms.
    - Inclusion of larger and more relevant datasets.
    - Improved feature engineering.
    - Optimized hyperparameters.

- **UI/UX Enhancements:**
  - Contributions towards improving the user interface and experience are highly appreciated. This includes design enhancements, usability improvements, and creative additions.

## Note to Contributors:
This project welcomes contributions to address model inefficiencies and improve user experience. Please feel free to fork the repository, make pull requests, and collaborate to make ASADWI a more robust and user-friendly platform. We appreciate efforts towards refining machine learning models, leveraging extensive datasets, and enhancing UI/UX for impactful spatial analysis. Let's create something valuable together!
