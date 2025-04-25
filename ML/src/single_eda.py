import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directories
os.makedirs('outputs/eda', exist_ok=True)
os.makedirs('outputs/eda/integrated', exist_ok=True)

# Water Quality functions
def load_water_quality_data(file_path='data/water_quality_data.csv'):
    """Load water quality data and perform basic cleaning"""
    print("Loading water quality data...")
    try:
        df = pd.read_csv(file_path)
        
        # Extract Maharashtra data
        maharashtra_df = df[df['State'] == 'Maharashtra'].copy()
        
        # Convert columns to numeric
        for col in maharashtra_df.columns:
            if col not in ['State', 'District', 'Block']:
                maharashtra_df[col] = pd.to_numeric(maharashtra_df[col], errors='coerce')
        
        # Handle missing values
        numeric_cols = maharashtra_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            maharashtra_df[col] = maharashtra_df[col].fillna(maharashtra_df[col].median())
        
        print(f"Loaded {len(maharashtra_df)} samples from Maharashtra")
        return maharashtra_df
    except Exception as e:
        print(f"Error loading water quality data: {e}")
        return None

def generate_water_quality_overview(df):
    """Generate overview statistics and distribution plots for water quality parameters"""
    print("Generating water quality overview...")
    
    # Key water quality parameters - ensure they exist in the dataframe
    available_params = [col for col in ['pH', 'TDS', 'EC in μS/cm', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'NO3'] 
                        if col in df.columns]
    
    # Calculate TDS threshold based on WHO guidelines (500 mg/L)
    if 'TDS' in df.columns:
        df['Water Safe'] = (df['TDS'] <= 500).astype(int)
    else:
        print("Warning: TDS column not found, cannot calculate water safety")
        df['Water Safe'] = 0  # Default to unsafe if TDS not available
    
    # Create summary statistics
    summary = df[available_params].describe().T
    summary['missing_pct'] = df[available_params].isnull().mean() * 100
    summary.to_csv('outputs/eda/water_quality_summary.csv')
    
    # Generate district-wise water quality map
    if 'District' in df.columns:
        district_safety = df.groupby('District')['Water Safe'].mean().sort_values()
        
        # Create district water quality plot
        plt.figure(figsize=(14, 8))
        colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
        norm = plt.Normalize(district_safety.min(), district_safety.max())
        colors = plt.cm.RdYlGn(norm(district_safety.values))
        
        ax = district_safety.plot(kind='bar', color=colors)
        ax.set_ylabel('Proportion of Safe Water Samples')
        ax.set_xlabel('District')
        ax.set_title('Proportion of Safe Water Samples by District in Maharashtra', fontsize=16)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Threshold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        plt.savefig('outputs/eda/district_water_safety.png')
        plt.close()
    else:
        print("Warning: District column not found, skipping district-level analysis")
    
    # Distribution of key parameters with safety thresholds
    thresholds = {
        'pH': (6.5, 8.5),  # WHO range
        'TDS': (500, None),  # WHO upper limit
        'EC in μS/cm': (750, None),  # Approximate upper limit
        'TH': (300, None),  # Hardness upper limit
        'F': (1.5, None),  # Fluoride upper limit
        'NO3': (45, None)   # Nitrate upper limit
    }
    
    # Only include thresholds for columns that exist in the dataframe
    available_thresholds = {param: thresholds[param] for param in thresholds if param in df.columns}
    
    # Create distribution plots with thresholds
    for param, (lower, upper) in available_thresholds.items():
        plt.figure(figsize=(12, 6))
        
        # Plot distribution and kernel density
        sns.histplot(df[param], kde=True, color='skyblue')
        
        # Add threshold lines
        if lower is not None:
            plt.axvline(x=lower, color='red', linestyle='--', 
                       label=f'Lower Limit: {lower}')
            
        if upper is not None:
            plt.axvline(x=upper, color='darkred', linestyle='--', 
                       label=f'Upper Limit: {upper}')
            
        # Set titles and labels
        plt.title(f'Distribution of {param} across Maharashtra', fontsize=14)
        plt.xlabel(param)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        # Fix: Create a safe filename without special characters
        safe_param = param.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "_").replace("*", "_").replace("<", "_").replace(">", "_").replace("|", "_").replace("μ", "u")
        plt.savefig(f'outputs/eda/distribution_{safe_param}.png')
        plt.close()
    
    # Save summary of parameter exceedances
    if 'District' in df.columns:
        exceedance_summary = pd.DataFrame(index=df['District'].unique())
        
        for param, (lower, upper) in available_thresholds.items():
            if lower is not None:
                col_name = f'{param}_below_{lower}'
                district_exceedance = df[df[param] < lower].groupby('District').size() / df.groupby('District').size()
                exceedance_summary[col_name] = district_exceedance
            
            if upper is not None:
                col_name = f'{param}_above_{upper}'
                district_exceedance = df[df[param] > upper].groupby('District').size() / df.groupby('District').size()
                exceedance_summary[col_name] = district_exceedance
        
        exceedance_summary.to_csv('outputs/eda/parameter_exceedance_by_district.csv')
    else:
        exceedance_summary = None
        print("Warning: District column not found, skipping exceedance summary by district")
    
    print("Water quality overview generated successfully")
    return exceedance_summary

def analyze_water_safety_factors(df):
    """Analyze factors contributing to water safety/potability"""
    print("Analyzing water safety factors...")
    
    # Key water quality parameters
    key_params = ['pH', 'TDS', 'EC in μS/cm', 'HCO3', 'Cl', 'TH', 'Ca', 'Mg', 'Na', 'K', 'F', 'NO3']
    available_params = [p for p in key_params if p in df.columns]
    
    if not available_params:
        print("Warning: No water quality parameters found in the dataframe")
        return None
    
    # Ensure Water Safe column exists
    if 'Water Safe' not in df.columns:
        if 'TDS' in df.columns:
            df['Water Safe'] = (df['TDS'] <= 500).astype(int)
        else:
            print("Warning: TDS column not found, cannot calculate water safety")
            df['Water Safe'] = 0  # Default to unsafe if TDS not available
    
    # Create correlation matrix
    analysis_cols = available_params + ['Water Safe']
    corr_matrix = df[analysis_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={'shrink': .7})
    plt.title('Parameter Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/eda/correlation_heatmap.png')
    plt.close()
    
    # Calculate correlation with water safety
    if 'Water Safe' in corr_matrix.columns:
        safety_corr = corr_matrix['Water Safe'].drop('Water Safe').sort_values(ascending=False)
        
        # Plot correlation with water safety
        plt.figure(figsize=(12, 8))
        colors = ['green' if x >= 0 else 'red' for x in safety_corr]
        safety_corr.plot(kind='bar', color=colors)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Correlation of Parameters with Water Safety', fontsize=16)
        plt.xlabel('Water Quality Parameter')
        plt.ylabel('Correlation Coefficient')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/eda/water_safety_correlation.png')
        plt.close()
    else:
        print("Warning: Water Safe column not in correlation matrix")
        safety_corr = None
    
    # Feature importance visualization using PCA
    if len(available_params) > 1:  # Need at least 2 features for PCA
        scaled_features = StandardScaler().fit_transform(df[available_params])
        pca = PCA()
        pca.fit(scaled_features)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=0.8, color='r', linestyle='-', alpha=0.5, label='80% Threshold')
        plt.title('Explained Variance by Components', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('outputs/eda/pca_explained_variance.png')
        plt.close()
        
        # Plot parameter contributions to principal components
        plt.figure(figsize=(14, 8))
        components = pd.DataFrame(pca.components_, columns=available_params)
        n_components = min(4, len(components))  # Show at most 4 components
        sns.heatmap(components.iloc[:n_components, :], annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Parameter Contributions to Principal Components', fontsize=16)
        plt.ylabel('Principal Components')
        plt.tight_layout()
        plt.savefig('outputs/eda/pca_components_heatmap.png')
        plt.close()
    else:
        print("Warning: Not enough features for PCA analysis")
    
    print("Water safety factors analyzed successfully")
    return safety_corr

def generate_water_quality_maps(df):
    """Generate geographical insights for water quality parameters"""
    print("Generating geographical water quality insights...")
    
    if 'District' not in df.columns:
        print("Warning: District column not found, skipping geographical analysis")
        return
    
    # Prepare district-level summaries
    available_params = [col for col in ['pH', 'TDS', 'EC in μS/cm', 'TH', 'F', 'NO3'] if col in df.columns]
    
    if not available_params:
        print("Warning: No water quality parameters found for district summary")
        return
    
    # Add Water Safe if not already present
    if 'Water Safe' not in df.columns and 'TDS' in df.columns:
        df['Water Safe'] = (df['TDS'] <= 500).astype(int)
    
    # Create summary with available parameters
    agg_dict = {param: 'mean' for param in available_params}
    if 'Water Safe' in df.columns:
        agg_dict['Water Safe'] = 'mean'
    
    district_summary = df.groupby('District').agg(agg_dict)
    
    # Format the numbers
    for col in district_summary.columns:
        if col == 'Water Safe':
            district_summary[col] = (district_summary[col] * 100).round(1)
        else:
            district_summary[col] = district_summary[col].round(2)
    
    # Save district summary
    district_summary.to_csv('outputs/eda/district_water_quality_summary.csv')
    
    # Create a choropleth-style bar chart for key parameters
    parameters = [param for param in ['pH', 'TDS', 'TH', 'F', 'NO3', 'Water Safe'] if param in district_summary.columns]
    
    for param in parameters:
        plt.figure(figsize=(14, 8))
        
        # Sort values
        sorted_data = district_summary[param].sort_values(ascending=False)
        
        # Create colormap based on parameter
        if param == 'Water Safe':
            # For water safety, higher is better
            norm = plt.Normalize(sorted_data.min(), sorted_data.max())
            colors = plt.cm.RdYlGn(norm(sorted_data.values))
            title = f'Water Safety Rate by District (%)'
        else:
            # For other parameters, use a neutral color scheme
            norm = plt.Normalize(sorted_data.min(), sorted_data.max())
            colors = plt.cm.viridis(norm(sorted_data.values))
            title = f'Average {param} by District'
        
        # Create bar chart
        bars = plt.bar(sorted_data.index, sorted_data, color=colors)
        plt.title(title, fontsize=16)
        plt.xlabel('District')
        plt.ylabel(param)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Fix: Create a safe filename without special characters
        safe_param = param.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_").replace("?", "_").replace("*", "_").replace("<", "_").replace(">", "_").replace("|", "_").replace("μ", "u")
        plt.savefig(f'outputs/eda/district_{safe_param}.png')
        plt.close()
    
    print("Geographical water quality insights generated successfully")
    return district_summary

# Geology and Lithology functions
def load_geology_data(file_path='data/Geology Maharashtra.xlsx'):
    """Load geology data and perform basic cleaning"""
    print("Loading geology data...")
    try:
        df = pd.read_excel(file_path)
        
        # Convert columns to numeric
        for col in df.columns:
            if col not in ['Location', 'District', 'Taluka', 'Village']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        print(f"Loaded {len(df)} geology samples")
        return df
    except Exception as e:
        print(f"Error loading geology data: {e}")
        return None

def load_lithology_data(file_path='data/Lithology Maharashtra.xlsx'):
    """Load lithology data and perform basic cleaning"""
    print("Loading lithology data...")
    try:
        df = pd.read_excel(file_path)
        
        # Check if SoilType column exists
        if 'SoilType' not in df.columns:
            soil_cols = [col for col in df.columns if 'soil' in col.lower() or 'type' in col.lower()]
            if soil_cols:
                df.rename(columns={soil_cols[0]: 'SoilType'}, inplace=True)
            else:
                print("Warning: SoilType column not found")
        
        # Ensure numeric columns are numeric
        for col in df.columns:
            if col not in ['SoilType', 'Location', 'District', 'Taluka', 'Village']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.fillna(0)
        
        print(f"Loaded {len(df)} lithology samples")
        return df
    except Exception as e:
        print(f"Error loading lithology data: {e}")
        return None

def analyze_geology_trends(geo_df):
    """Analyze geology trends and generate insights"""
    print("Analyzing geology trends...")
    
    # Key parameters
    key_params = ['Depth', 'LaDeg', 'LaMin', 'LaSec', 'LoDeg', 'LoMin', 'LoSec', 'Elevation']
    available_params = [p for p in key_params if p in geo_df.columns]
    
    # Distribution of well depths
    if 'Depth' in geo_df.columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(geo_df['Depth'], kde=True, bins=30, color='steelblue')
        plt.axvline(x=geo_df['Depth'].median(), color='red', linestyle='--', 
                   label=f'Median: {geo_df["Depth"].median():.1f}m')
        plt.axvline(x=geo_df['Depth'].mean(), color='green', linestyle='--', 
                   label=f'Mean: {geo_df["Depth"].mean():.1f}m')
        plt.title('Distribution of Well Depths in Maharashtra', fontsize=16)
        plt.xlabel('Depth (meters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/eda/well_depth_distribution.png')
        plt.close()
        
        # District-wise depth analysis
        if 'District' in geo_df.columns:
            district_depth = geo_df.groupby('District')['Depth'].agg(['mean', 'median', 'min', 'max', 'count'])
            district_depth = district_depth.sort_values('median')
            
            plt.figure(figsize=(14, 8))
            ax = sns.boxplot(x='District', y='Depth', data=geo_df, order=district_depth.index)
            plt.xticks(rotation=45, ha='right')
            plt.title('Well Depth Distribution by District', fontsize=16)
            plt.xlabel('District')
            plt.ylabel('Depth (meters)')
            plt.axhline(y=geo_df['Depth'].median(), color='red', linestyle='--', alpha=0.7)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/district_depth_boxplot.png')
            plt.close()
            
            # Save district depth summary
            district_depth.to_csv('outputs/eda/district_depth_summary.csv')
    
    # Create correlation matrix for geological parameters
    if len(available_params) > 1:
        corr_matrix = geo_df[available_params].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={'shrink': .7})
        plt.title('Correlation of Geological Parameters', fontsize=16)
        plt.tight_layout()
        plt.savefig('outputs/eda/geology_correlation_matrix.png')
        plt.close()
    
    print("Geology trends analyzed successfully")
    return

def analyze_lithology_patterns(litho_df):
    """Analyze lithology patterns and generate insights"""
    print("Analyzing lithology patterns...")
    
    # Check if we have soil type column
    if 'SoilType' not in litho_df.columns:
        print("SoilType column not found in lithology data")
        return
    
    # Soil type distribution
    plt.figure(figsize=(14, 8))
    soil_counts = litho_df['SoilType'].value_counts()
    ax = soil_counts.plot(kind='bar')
    plt.title('Distribution of Soil Types in Maharashtra', fontsize=16)
    plt.xlabel('Soil Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for i, count in enumerate(soil_counts):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/eda/soil_type_distribution.png')
    plt.close()
    
    print("Lithology patterns analyzed successfully")
    return

# Integrated analysis functions
def integrated_cross_domain_analysis(water_df, geo_df, litho_df):
    """Perform integrated analysis across water quality, geology, and lithology domains"""
    print("Performing integrated cross-domain analysis...")
    
    # Check if we have district in all dataframes
    valid_dfs = [df for df in [water_df, geo_df, litho_df] if df is not None]
    if all('District' in df.columns for df in valid_dfs):
        # Create district-level summary dataframes
        dfs = []
        
        if water_df is not None and 'TDS' in water_df.columns:
            # Water quality summary by district
            if 'Water Safe' not in water_df.columns:
                water_df['Water Safe'] = (water_df['TDS'] <= 500).astype(int)
            
            available_cols = [col for col in ['pH', 'TDS'] if col in water_df.columns]
            if available_cols:
                agg_dict = {col: 'mean' for col in available_cols}
                agg_dict['Water Safe'] = 'mean'
                
                water_summary = water_df.groupby('District').agg(agg_dict)
                
                # Rename columns to avoid conflicts
                rename_dict = {col: col for col in available_cols}
                rename_dict['Water Safe'] = 'WaterSafetyRate'
                water_summary.rename(columns=rename_dict, inplace=True)
                
                dfs.append(water_summary)
        
        if geo_df is not None and 'Depth' in geo_df.columns:
            # Geology summary by district
            geo_summary = geo_df.groupby('District').agg({
                'Depth': 'median'
            })
            geo_summary.columns = ['MedianWellDepth']
            dfs.append(geo_summary)
        
        if litho_df is not None and 'SoilType' in litho_df.columns and 'District' in litho_df.columns:
            # Get most common soil type by district
            try:
                soil_counts = pd.crosstab(litho_df['District'], litho_df['SoilType'])
                dominant_soil = soil_counts.idxmax(axis=1).to_frame('DominantSoilType')
                dfs.append(dominant_soil)
            except Exception as e:
                print(f"Warning: Error computing dominant soil type: {e}")
        
        if dfs:
            try:
                # Merge all available district summaries
                from functools import reduce
                district_integrated = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
                
                # Fill missing values for numeric columns
                for col in district_integrated.columns:
                    if district_integrated[col].dtype in [np.float64, np.int64]:
                        district_integrated[col] = district_integrated[col].fillna(district_integrated[col].mean())
                
                # Save integrated district data
                district_integrated.to_csv('outputs/eda/integrated/district_integrated_summary.csv')
                
                # Create visualization showing relationship between depth and water quality
                if 'MedianWellDepth' in district_integrated.columns and 'WaterSafetyRate' in district_integrated.columns:
                    plt.figure(figsize=(12, 8))
                    
                    # Create scatter plot with color based on water safety
                    norm = plt.Normalize(0, 1)
                    scatter = plt.scatter(district_integrated['MedianWellDepth'], 
                                        district_integrated['WaterSafetyRate'] * 100,
                                        c=district_integrated['WaterSafetyRate'],
                                        cmap='RdYlGn', s=100, alpha=0.7, norm=norm)
                    
                    # Add district labels
                    for i, district in enumerate(district_integrated.index):
                        plt.annotate(district, 
                                    (district_integrated['MedianWellDepth'].iloc[i], 
                                    district_integrated['WaterSafetyRate'].iloc[i] * 100),
                                    xytext=(5, 5), textcoords='offset points')
                    
                    plt.colorbar(scatter, label='Water Safety Rate (0-1)')
                    plt.title('Relationship Between Well Depth and Water Safety by District', fontsize=16)
                    plt.xlabel('Median Well Depth (meters)')
                    plt.ylabel('Water Safety Rate (%)')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('outputs/eda/integrated/depth_water_safety_relationship.png')
                    plt.close()
            except Exception as e:
                print(f"Warning: Error in integrated analysis: {e}")
    else:
        print("Warning: Not all dataframes have District column, skipping cross-domain analysis")
    
    print("Integrated cross-domain analysis completed")
    return

def main():
    """Main function to run the entire EDA process"""
    print("\nStarting Comprehensive EDA...\n")
    
    # Load all datasets
    water_data = load_water_quality_data()
    geology_data = load_geology_data()
    lithology_data = load_lithology_data()
    
    # Run water quality EDA
    if water_data is not None:
        generate_water_quality_overview(water_data)
        analyze_water_safety_factors(water_data)
        generate_water_quality_maps(water_data)
    
    # Run geology and lithology EDA
    if geology_data is not None:
        analyze_geology_trends(geology_data)
    
    if lithology_data is not None:
        analyze_lithology_patterns(lithology_data)
    
    # Run integrated cross-domain analysis
    if any(data is not None for data in [water_data, geology_data, lithology_data]):
        integrated_cross_domain_analysis(water_data, geology_data, lithology_data)
    
    print("\nComprehensive EDA completed successfully! Output files saved to outputs/eda/")
    print("These visualizations are ready for your presentation to the professor.")

if __name__ == "__main__":
    main() 