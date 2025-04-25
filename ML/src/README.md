# AquaSense Exploratory Data Analysis (EDA)

This directory contains Python scripts for comprehensive exploratory data analysis of water quality, geology, and lithology data for the AquaSense project.

## Purpose

These EDA scripts are designed to extract meaningful and practical insights from the data that can be used in presentations to demonstrate the value of the AquaSense project. The analyses focus on:

1. **Water Quality Analysis**: Understanding the distribution of water quality parameters, identifying unsafe water regions, and determining key factors affecting water potability.

2. **Geology Analysis**: Examining well depth distributions, geological clusters, and the relationship between elevation and well depth.

3. **Lithology Analysis**: Analyzing soil type distributions, layer thickness patterns, and district-specific soil compositions.

4. **Integrated Analysis**: Combining all three domains to understand how geological and lithological factors influence water quality.

## Scripts Overview

- **water_quality_eda.py**: Focuses on water quality parameters, safety thresholds, and geographical distribution of water safety.

- **geology_lithology_eda.py**: Analyzes geology and lithology data, including well depths, soil types, and their relationships.

- **integrated_eda.py**: Runs both previous scripts and performs cross-domain analysis, generating presentation-ready visualizations.

## Running the Scripts

### Prerequisites

Make sure you have the following Python packages installed:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install them using:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execution

To run the complete analysis:

```
python integrated_eda.py
```

To run individual analyses:

```
python water_quality_eda.py
python geology_lithology_eda.py
```

## Output

All visualizations and data summaries are saved to the `../outputs/eda/` directory, organized as follows:

- `../outputs/eda/`: Individual domain visualizations
- `../outputs/eda/integrated/`: Cross-domain analyses and presentation-ready visualizations

### Key Outputs for Presentation

1. **Executive Summaries**: High-quality dashboard-style visualizations suitable for presentations

2. **District Quadrant Analysis**: Shows the relationship between well depth and water safety by district

3. **Key Findings Summary**: A markdown file summarizing the most important insights

4. **Parameter Correlation Heatmaps**: Shows how different water quality and geological parameters relate to each other

5. **Geographical Insights**: Maps and district-level summaries of water quality and geological characteristics

## Data Interpretation Guide

When presenting the results, focus on these key aspects:

### Water Quality

- Districts with the highest/lowest water safety rates
- Parameters most strongly correlated with water safety
- Geographical patterns in water quality parameters

### Geology and Lithology

- Relationship between well depth and water safety
- Impact of soil types on water quality
- Optimal well depth ranges for different regions

### Actionable Insights

- Priority districts for intervention
- Recommended well depths for new constructions
- Soil types that indicate potential water quality issues

## Customization

The scripts can be customized by modifying the threshold values, parameters of interest, or adding new analyses as needed. Look for comments in the code indicating where customizations can be made.

---

**Note**: This EDA is designed to be practical and presentation-focused, emphasizing insights that demonstrate the value of the AquaSense AI system for water quality prediction and well planning. 