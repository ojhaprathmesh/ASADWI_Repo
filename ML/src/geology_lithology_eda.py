import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directory
os.makedirs('outputs/eda', exist_ok=True)

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
    
    # Correlation between elevation and depth
    if 'Elevation' in geo_df.columns and 'Depth' in geo_df.columns:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Elevation', y='Depth', data=geo_df, alpha=0.7)
        
        # Add trend line
        sns.regplot(x='Elevation', y='Depth', data=geo_df, scatter=False, color='red')
        
        plt.title('Relationship Between Elevation and Well Depth', fontsize=16)
        plt.xlabel('Elevation (meters)')
        plt.ylabel('Depth (meters)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/eda/elevation_depth_relationship.png')
        plt.close()
        
        # Calculate and print correlation
        corr = geo_df[['Elevation', 'Depth']].corr().iloc[0,1]
        print(f"Correlation between Elevation and Depth: {corr:.3f}")
    
    # Create correlation matrix for geological parameters
    if available_params:
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
    
    # Clustering analysis for geological zones
    if len(available_params) >= 3:
        # Select relevant features for clustering
        features = [p for p in available_params if p != 'Depth'][:3]  # Use up to 3 features
        
        if len(features) >= 2:
            # Scale the features
            X = geo_df[features].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)
            
            # Plot elbow curve
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
            plt.title('Elbow Method for Optimal Clusters', fontsize=16)
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/geology_elbow_curve.png')
            plt.close()
            
            # Choose optimal clusters (for example, 4)
            optimal_clusters = 4  # This would ideally be determined from the elbow curve
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            geo_df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Visualize clusters
            if len(features) >= 2:
                plt.figure(figsize=(12, 10))
                
                # Create scatter plot with the first two features
                scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                     c=geo_df['Cluster'], cmap='viridis', s=50, alpha=0.7)
                
                # Add cluster centers
                centers = kmeans.cluster_centers_
                plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
                
                plt.title('Geological Clusters in Maharashtra', fontsize=16)
                plt.xlabel(features[0])
                plt.ylabel(features[1])
                plt.colorbar(scatter, label='Cluster')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('outputs/eda/geology_clusters.png')
                plt.close()
                
                # Analyze clusters
                cluster_analysis = geo_df.groupby('Cluster').agg({
                    'Depth': ['mean', 'median', 'min', 'max', 'count']
                })
                
                # Plot depth by cluster
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='Cluster', y='Depth', data=geo_df)
                plt.title('Well Depth Distribution by Geological Cluster', fontsize=16)
                plt.xlabel('Cluster')
                plt.ylabel('Depth (meters)')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('outputs/eda/cluster_depth_boxplot.png')
                plt.close()
                
                # Save cluster analysis
                cluster_analysis.to_csv('outputs/eda/geological_cluster_analysis.csv')
    
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
    
    # Analyze soil layers by depth
    if all(col in litho_df.columns for col in ['From', 'To']):
        # Create depth ranges
        bins = [0, 10, 20, 50, 100, float('inf')]
        labels = ['0-10m', '10-20m', '20-50m', '50-100m', '100m+']
        litho_df['DepthRange'] = pd.cut(litho_df['To'], bins=bins, labels=labels)
        
        # Count soil types by depth range
        depth_soil_pivot = pd.crosstab(litho_df['DepthRange'], litho_df['SoilType'])
        
        # Plot soil type by depth
        plt.figure(figsize=(16, 10))
        depth_soil_pivot.plot(kind='bar', stacked=True, colormap='tab20')
        plt.title('Soil Type Distribution by Depth Range', fontsize=16)
        plt.xlabel('Depth Range')
        plt.ylabel('Frequency')
        plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/eda/soil_type_by_depth.png')
        plt.close()
        
        # Analyze average thickness by soil type
        if 'Thickness' in litho_df.columns:
            thickness_by_soil = litho_df.groupby('SoilType')['Thickness'].agg(['mean', 'median', 'min', 'max', 'count'])
            thickness_by_soil = thickness_by_soil.sort_values('mean', ascending=False)
            
            plt.figure(figsize=(14, 8))
            ax = thickness_by_soil['mean'].plot(kind='bar')
            plt.title('Average Layer Thickness by Soil Type', fontsize=16)
            plt.xlabel('Soil Type')
            plt.ylabel('Average Thickness (meters)')
            plt.xticks(rotation=45, ha='right')
            
            # Add thickness labels
            for i, thickness in enumerate(thickness_by_soil['mean']):
                ax.text(i, thickness + 0.1, f"{thickness:.1f}m", ha='center')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/soil_type_thickness.png')
            plt.close()
            
            # Save soil type thickness summary
            thickness_by_soil.to_csv('outputs/eda/soil_type_thickness.csv')
        
        # Create soil type profiles by district
        if 'District' in litho_df.columns:
            district_soil_pivot = pd.crosstab(litho_df['District'], litho_df['SoilType'], normalize='index')
            district_soil_pivot = district_soil_pivot.sort_index()
            
            plt.figure(figsize=(16, 10))
            district_soil_pivot.plot(kind='bar', stacked=True, colormap='tab20')
            plt.title('Soil Type Composition by District', fontsize=16)
            plt.xlabel('District')
            plt.ylabel('Proportion')
            plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/district_soil_composition.png')
            plt.close()
            
            # Save district soil composition
            district_soil_pivot.to_csv('outputs/eda/district_soil_composition.csv')
    
    print("Lithology patterns analyzed successfully")
    return

def integrate_geology_lithology(geo_df, litho_df):
    """Integrate geology and lithology data for comprehensive insights"""
    print("Integrating geology and lithology data...")
    
    # Check if we have district in both dataframes
    if 'District' in geo_df.columns and 'District' in litho_df.columns:
        # Get district depth summaries
        district_depth = geo_df.groupby('District')['Depth'].median().to_frame()
        
        # Get district soil type composition
        district_soil = pd.crosstab(litho_df['District'], litho_df['SoilType'])
        district_soil_pct = district_soil.div(district_soil.sum(axis=1), axis=0) * 100
        
        # Merge the data
        common_districts = set(district_depth.index) & set(district_soil_pct.index)
        
        if common_districts:
            # Filter to common districts
            district_depth = district_depth.loc[common_districts]
            district_soil_pct = district_soil_pct.loc[common_districts]
            
            # Get most common soil type per district
            district_soil_pct['MostCommonSoil'] = district_soil_pct.idxmax(axis=1)
            
            # Merge depth and soil data
            district_integrated = district_depth.merge(district_soil_pct[['MostCommonSoil']], 
                                                     left_index=True, right_index=True)
            
            # Plot depth by most common soil type
            plt.figure(figsize=(14, 8))
            
            # Group by soil type and calculate mean depth
            soil_depth = district_integrated.groupby('MostCommonSoil')['Depth'].mean().sort_values()
            
            ax = soil_depth.plot(kind='bar', colormap='viridis')
            plt.title('Average Well Depth by Predominant Soil Type', fontsize=16)
            plt.xlabel('Predominant Soil Type')
            plt.ylabel('Average Depth (meters)')
            plt.xticks(rotation=45, ha='right')
            
            # Add depth labels
            for i, depth in enumerate(soil_depth):
                ax.text(i, depth + 0.5, f"{depth:.1f}m", ha='center')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/soil_type_depth_relationship.png')
            plt.close()
            
            # Save integrated district data
            district_integrated.to_csv('outputs/eda/district_integrated_analysis.csv')
    
    print("Integration analysis completed")
    return

def main():
    # Load geology data
    geology_data = load_geology_data()
    
    # Load lithology data
    lithology_data = load_lithology_data()
    
    if geology_data is not None:
        # Analyze geology trends
        analyze_geology_trends(geology_data)
    
    if lithology_data is not None:
        # Analyze lithology patterns
        analyze_lithology_patterns(lithology_data)
    
    if geology_data is not None and lithology_data is not None:
        # Integrate geology and lithology data
        integrate_geology_lithology(geology_data, lithology_data)
    
    print("\nEDA completed successfully! Output files saved to outputs/eda/")
    print("These visualizations provide meaningful insights for your presentation.")

if __name__ == "__main__":
    main() 