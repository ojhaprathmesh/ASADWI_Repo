import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

# Import our EDA modules
from water_quality_eda import load_water_quality_data, generate_water_quality_overview, analyze_water_safety_factors, generate_water_quality_maps
from geology_lithology_eda import load_geology_data, load_lithology_data, analyze_geology_trends, analyze_lithology_patterns, integrate_geology_lithology

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette('deep')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# Create output directory
os.makedirs('outputs/eda', exist_ok=True)
os.makedirs('outputs/eda/integrated', exist_ok=True)

def integrated_cross_domain_analysis(water_df, geo_df, litho_df):
    """Perform integrated analysis across water quality, geology, and lithology domains"""
    print("Performing integrated cross-domain analysis...")
    
    # Check if we have district in all dataframes
    if all('District' in df.columns for df in [water_df, geo_df, litho_df] if df is not None):
        # Create district-level summary dataframes
        dfs = []
        
        if water_df is not None:
            # Water quality summary by district
            water_df['Water Safe'] = (water_df['TDS'] <= 500).astype(int)
            water_summary = water_df.groupby('District').agg({
                'pH': 'mean',
                'TDS': 'mean',
                'Water Safe': 'mean'
            })
            water_summary.columns = ['pH', 'TDS', 'WaterSafetyRate']
            dfs.append(water_summary)
        
        if geo_df is not None:
            # Geology summary by district
            geo_summary = geo_df.groupby('District').agg({
                'Depth': 'median'
            })
            geo_summary.columns = ['MedianWellDepth']
            dfs.append(geo_summary)
        
        if litho_df is not None and 'SoilType' in litho_df.columns:
            # Get most common soil type by district
            soil_counts = pd.crosstab(litho_df['District'], litho_df['SoilType'])
            dominant_soil = soil_counts.idxmax(axis=1).to_frame('DominantSoilType')
            dfs.append(dominant_soil)
        
        if dfs:
            # Merge all available district summaries
            from functools import reduce
            district_integrated = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
            
            # Fill missing values
            district_integrated = district_integrated.fillna({
                'pH': district_integrated['pH'].mean() if 'pH' in district_integrated.columns else np.nan,
                'TDS': district_integrated['TDS'].mean() if 'TDS' in district_integrated.columns else np.nan,
                'WaterSafetyRate': district_integrated['WaterSafetyRate'].mean() if 'WaterSafetyRate' in district_integrated.columns else np.nan,
                'MedianWellDepth': district_integrated['MedianWellDepth'].mean() if 'MedianWellDepth' in district_integrated.columns else np.nan
            })
            
            # Save integrated district data
            district_integrated.to_csv('outputs/eda/integrated/district_integrated_summary.csv')
            
            # Create visualization showing relationship between depth and water quality
            if 'MedianWellDepth' in district_integrated.columns and 'WaterSafetyRate' in district_integrated.columns:
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot with color based on water safety
                scatter = plt.scatter(district_integrated['MedianWellDepth'], 
                                      district_integrated['WaterSafetyRate'] * 100,
                                      c=district_integrated['WaterSafetyRate'],
                                      cmap='RdYlGn', s=100, alpha=0.7)
                
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
                
                # If we have dominant soil type, analyze its relationship with water safety
                if 'DominantSoilType' in district_integrated.columns:
                    plt.figure(figsize=(14, 8))
                    soil_water_safety = district_integrated.groupby('DominantSoilType')['WaterSafetyRate'].mean().sort_values()
                    
                    # Create colormap based on water safety
                    colors = sns.color_palette("RdYlGn", len(soil_water_safety))
                    
                    # Create bar chart
                    ax = soil_water_safety.plot(kind='bar', color=colors)
                    plt.title('Water Safety by Dominant Soil Type', fontsize=16)
                    plt.xlabel('Dominant Soil Type')
                    plt.ylabel('Average Water Safety Rate')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add percentage labels
                    for i, rate in enumerate(soil_water_safety):
                        ax.text(i, rate + 0.01, f"{rate:.1%}", ha='center')
                    
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('outputs/eda/integrated/soil_type_water_safety.png')
                    plt.close()
    
    # Create integrated data table summarizing key findings
    summary_points = [
        "# Integrated Water Quality, Geology, and Lithology Analysis",
        "\n## Key Findings\n"
    ]
    
    # Add key water quality findings
    if water_df is not None:
        safe_rate = water_df['Water Safe'].mean() if 'Water Safe' in water_df.columns else None
        if safe_rate is not None:
            summary_points.append(f"- **Water Safety Rate:** {safe_rate:.1%} of water samples in Maharashtra are safe for drinking")
        
        if 'TDS' in water_df.columns:
            summary_points.append(f"- **Average TDS:** {water_df['TDS'].mean():.1f} mg/L (WHO limit: 500 mg/L)")
        
        if 'pH' in water_df.columns:
            summary_points.append(f"- **Average pH:** {water_df['pH'].mean():.1f} (Ideal range: 6.5-8.5)")
    
    # Add key geology findings
    if geo_df is not None and 'Depth' in geo_df.columns:
        summary_points.append(f"- **Average Well Depth:** {geo_df['Depth'].mean():.1f} meters")
        summary_points.append(f"- **Median Well Depth:** {geo_df['Depth'].median():.1f} meters")
    
    # Add key lithology findings
    if litho_df is not None and 'SoilType' in litho_df.columns:
        most_common_soil = litho_df['SoilType'].value_counts().idxmax()
        soil_percentage = litho_df['SoilType'].value_counts().max() / len(litho_df) * 100
        summary_points.append(f"- **Most Common Soil Type:** {most_common_soil} ({soil_percentage:.1f}% of samples)")
    
    # Add correlations between domains if available
    if 'district_integrated' in locals() and 'MedianWellDepth' in district_integrated.columns and 'WaterSafetyRate' in district_integrated.columns:
        corr = district_integrated[['MedianWellDepth', 'WaterSafetyRate']].corr().iloc[0,1]
        summary_points.append(f"- **Correlation between Well Depth and Water Safety:** {corr:.3f}")
    
    # Add section on recommended practices
    summary_points.extend([
        "\n## Recommended Practices\n",
        "- **Optimal Well Depth:** Based on the analysis, wells between X and Y meters deep tend to provide safer water",
        "- **Soil Considerations:** Wells in [specific soil type] areas should implement additional filtration due to higher TDS levels",
        "- **Geographical Focus:** Districts like [names] require immediate intervention due to poor water quality indicators",
        "- **Seasonal Awareness:** Water quality monitoring should be intensified during [season] when contamination risks increase"
    ])
    
    # Create markdown summary file
    with open('outputs/eda/integrated/key_findings_summary.md', 'w') as f:
        f.write('\n'.join(summary_points))
    
    print("Integrated cross-domain analysis completed")
    return

def create_presentation_ready_visualizations(water_df, geo_df, litho_df):
    """Create high-quality visualizations specifically designed for presentations"""
    print("Creating presentation-ready visualizations...")
    
    # Set presentation-friendly styling
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # Create water quality executive summary visualization
    if water_df is not None:
        # Ensure Water Safe column exists
        if 'Water Safe' not in water_df.columns:
            water_df['Water Safe'] = (water_df['TDS'] <= 500).astype(int)
        
        # Create 2x2 dashboard of key water metrics
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Water Quality Executive Summary - Maharashtra', fontsize=24, y=0.95)
        
        # Plot 1: Safe vs Unsafe pie chart
        safe_count = water_df['Water Safe'].sum()
        unsafe_count = len(water_df) - safe_count
        axes[0, 0].pie([safe_count, unsafe_count], 
                     labels=['Safe', 'Unsafe'], 
                     autopct='%1.1f%%',
                     colors=['#2ecc71', '#e74c3c'],
                     explode=(0.05, 0),
                     shadow=True,
                     startangle=90)
        axes[0, 0].set_title('Water Safety Distribution', fontsize=18)
        
        # Plot 2: Top 5 districts by water safety
        if 'District' in water_df.columns:
            district_safety = water_df.groupby('District')['Water Safe'].mean().sort_values(ascending=False)
            top_districts = district_safety.head(5)
            bottom_districts = district_safety.tail(5)
            
            # Plot top districts
            bars = axes[0, 1].bar(top_districts.index, top_districts * 100, color='#2ecc71')
            axes[0, 1].set_title('Top 5 Districts by Water Safety Rate', fontsize=18)
            axes[0, 1].set_ylabel('Water Safety Rate (%)')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].set_xticklabels(top_districts.index, rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{height:.1f}%', ha='center')
        
        # Plot 3: Distribution of key parameter (TDS)
        if 'TDS' in water_df.columns:
            sns.histplot(water_df['TDS'], bins=30, kde=True, color='#3498db', ax=axes[1, 0])
            axes[1, 0].axvline(x=500, color='red', linestyle='--', 
                             label='WHO Limit (500 mg/L)')
            axes[1, 0].set_title('TDS Distribution', fontsize=18)
            axes[1, 0].set_xlabel('Total Dissolved Solids (mg/L)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Plot 4: Bottom 5 districts by water safety
        if 'District' in water_df.columns and not bottom_districts.empty:
            bars = axes[1, 1].bar(bottom_districts.index, bottom_districts * 100, color='#e74c3c')
            axes[1, 1].set_title('Bottom 5 Districts by Water Safety Rate', fontsize=18)
            axes[1, 1].set_ylabel('Water Safety Rate (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].set_xticklabels(bottom_districts.index, rotation=45, ha='right')
            
            # Add percentage labels
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{height:.1f}%', ha='center')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('outputs/eda/integrated/water_quality_executive_summary.png')
        plt.close()
    
    # Create geology executive summary
    if geo_df is not None and 'Depth' in geo_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Geological Insights Executive Summary - Maharashtra', fontsize=24, y=0.95)
        
        # Plot 1: Well depth distribution
        sns.histplot(geo_df['Depth'], bins=30, kde=True, color='#3498db', ax=axes[0, 0])
        axes[0, 0].axvline(x=geo_df['Depth'].median(), color='red', linestyle='--', 
                         label=f'Median: {geo_df["Depth"].median():.1f}m')
        axes[0, 0].set_title('Well Depth Distribution', fontsize=18)
        axes[0, 0].set_xlabel('Depth (meters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: Depth categories pie chart
        depth_cats = pd.cut(geo_df['Depth'], 
                          bins=[0, 20, 50, 100, float('inf')],
                          labels=['Shallow (0-20m)', 'Medium (20-50m)', 'Deep (50-100m)', 'Very Deep (>100m)'])
        depth_counts = depth_cats.value_counts()
        
        colors = ['#1abc9c', '#3498db', '#9b59b6', '#34495e']
        axes[0, 1].pie(depth_counts, 
                     labels=depth_counts.index, 
                     autopct='%1.1f%%',
                     colors=colors,
                     explode=[0.05] * len(depth_counts),
                     shadow=True,
                     startangle=90)
        axes[0, 1].set_title('Well Depth Categories', fontsize=18)
        
        # Plot 3: Top districts by median depth
        if 'District' in geo_df.columns:
            district_depth = geo_df.groupby('District')['Depth'].median().sort_values(ascending=False)
            top_depth_districts = district_depth.head(5)
            
            bars = axes[1, 0].bar(top_depth_districts.index, top_depth_districts, 
                               color=plt.cm.viridis(np.linspace(0, 1, len(top_depth_districts))))
            axes[1, 0].set_title('Top 5 Districts by Well Depth', fontsize=18)
            axes[1, 0].set_ylabel('Median Depth (meters)')
            axes[1, 0].set_xticklabels(top_depth_districts.index, rotation=45, ha='right')
            
            # Add depth labels
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{height:.1f}m', ha='center')
        
        # Plot 4: Relationship with elevation if available
        if 'Elevation' in geo_df.columns:
            sns.scatterplot(x='Elevation', y='Depth', data=geo_df, 
                          alpha=0.7, color='#2980b9', ax=axes[1, 1])
            sns.regplot(x='Elevation', y='Depth', data=geo_df, 
                      scatter=False, color='red', ax=axes[1, 1])
            
            axes[1, 1].set_title('Relationship Between Elevation and Well Depth', fontsize=18)
            axes[1, 1].set_xlabel('Elevation (meters)')
            axes[1, 1].set_ylabel('Depth (meters)')
            
            # Add correlation coefficient
            corr = geo_df[['Elevation', 'Depth']].corr().iloc[0,1]
            axes[1, 1].annotate(f'Correlation: {corr:.3f}', 
                              xy=(0.05, 0.95), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('outputs/eda/integrated/geology_executive_summary.png')
        plt.close()
    
    # Create integrated insights visualization
    if all(df is not None for df in [water_df, geo_df, litho_df]):
        if all('District' in df.columns for df in [water_df, geo_df, litho_df]):
            # Create integrated district summaries
            water_district = water_df.groupby('District')['Water Safe'].mean().to_frame('Water Safety Rate')
            geo_district = geo_df.groupby('District')['Depth'].median().to_frame('Median Depth')
            
            # Merge data
            district_data = water_district.merge(geo_district, left_index=True, right_index=True)
            
            # Create scatter plot with quadrants
            plt.figure(figsize=(16, 12))
            
            # Plot scatter with district points
            scatter = plt.scatter(district_data['Median Depth'], 
                                district_data['Water Safety Rate'] * 100,
                                s=100, alpha=0.7,
                                c=district_data['Water Safety Rate'],
                                cmap='RdYlGn')
            
            # Add district labels
            for i, district in enumerate(district_data.index):
                plt.annotate(district, 
                           (district_data['Median Depth'].iloc[i], 
                            district_data['Water Safety Rate'].iloc[i] * 100),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12)
            
            # Add quadrant lines
            median_depth = district_data['Median Depth'].median()
            median_safety = 50  # 50% safety rate as threshold
            
            plt.axvline(x=median_depth, color='gray', linestyle='--', alpha=0.7)
            plt.axhline(y=median_safety, color='gray', linestyle='--', alpha=0.7)
            
            # Add quadrant labels
            plt.text(district_data['Median Depth'].max() * 0.95, 75, 
                   'Deep Wells\nSafe Water', 
                   ha='right', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#c8e6c9", alpha=0.8))
            
            plt.text(district_data['Median Depth'].min() * 1.05, 75, 
                   'Shallow Wells\nSafe Water', 
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#c8e6c9", alpha=0.8))
            
            plt.text(district_data['Median Depth'].max() * 0.95, 25, 
                   'Deep Wells\nUnsafe Water', 
                   ha='right', fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#ffccbc", alpha=0.8))
            
            plt.text(district_data['Median Depth'].min() * 1.05, 25, 
                   'Shallow Wells\nUnsafe Water', 
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", fc="#ffccbc", alpha=0.8))
            
            plt.colorbar(scatter, label='Water Safety Rate (0-1)')
            plt.title('Integrated Analysis: Well Depth vs. Water Safety by District', fontsize=22)
            plt.xlabel('Median Well Depth (meters)', fontsize=16)
            plt.ylabel('Water Safety Rate (%)', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('outputs/eda/integrated/district_quadrant_analysis.png')
            plt.close()
    
    print("Presentation-ready visualizations created successfully")
    return

def main():
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
    
    if geology_data is not None and lithology_data is not None:
        integrate_geology_lithology(geology_data, lithology_data)
    
    # Run integrated cross-domain analysis
    if any(data is not None for data in [water_data, geology_data, lithology_data]):
        integrated_cross_domain_analysis(water_data, geology_data, lithology_data)
        create_presentation_ready_visualizations(water_data, geology_data, lithology_data)
    
    print("\nComprehensive EDA completed successfully! Output files saved to outputs/eda/")
    print("These visualizations are ready for your presentation to the professor.")

if __name__ == "__main__":
    main() 