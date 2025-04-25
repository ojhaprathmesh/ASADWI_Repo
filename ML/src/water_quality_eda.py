import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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