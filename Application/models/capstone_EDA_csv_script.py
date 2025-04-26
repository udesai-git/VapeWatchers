from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch
import boto3   

    
def main(AWS_ACCESS_KEY,AWS_SECRET_KEY,aws_session_token):
    
    s3_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=aws_session_token
    )

    s3 = s3_session.client('s3')
    csv_obj = s3.get_object(Bucket='vapewatchers-2025', Key='vape_ocr_text_analysis.csv')
    body = csv_obj['Body'].read()

    df = pd.read_csv(BytesIO(body))
    # Define the static image folder
    visualization_folder = "static/visualization_images"

    # Delete old images
    if os.path.exists(visualization_folder):
        for file in os.listdir(visualization_folder):
            file_path = os.path.join(visualization_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(visualization_folder)

    # Basic dataset information
    print(f"Dataset shape: {df.shape}")
    print(f"Number of brands: {df['brand'].nunique()}")

    # Brand distribution
    brand_counts = df['brand'].value_counts().reset_index()
    brand_counts.columns = ['brand', 'count']
    top_brands = brand_counts[brand_counts['count'] >= 50]['brand'].tolist()

    # Filter to top brands for better visualization
    df_top_brands = df[df['brand'].isin(top_brands)]

    # Create a figure with subplots
    plt.figure(figsize=(10,10))
    # 1. Brand Distribution
    sns.barplot(x='count', y='brand', data=brand_counts.head(10), palette='viridis')
    plt.title('Top 10 Brands by Number of Images', fontsize=14)
    plt.xlabel('Number of Images')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Top_10_Brands_by_Number.png")
    plt.savefig(image_path)
    plt.close()

    # 2. Youth Appeal Score by Brand
    youth_appeal_by_brand = df_top_brands.groupby('brand')['youth_appeal_score'].mean().reset_index()
    youth_appeal_by_brand = youth_appeal_by_brand.sort_values('youth_appeal_score', ascending=False)

    plt.figure(figsize=(10,10))
    sns.barplot(x='youth_appeal_score', y='brand', data=youth_appeal_by_brand, palette='viridis')
    plt.title('Youth Appeal Score by Brand', fontsize=14)
    plt.xlabel('Average Youth Appeal Score')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Youth_Appeal_Score_by_Brand.png")
    plt.savefig(image_path)
    plt.close()

    # 3. Flavor and Slang Word Usage
    flavor_slang_by_brand = pd.DataFrame({
        'brand': df_top_brands['brand'].unique()
    })

    # Calculate average slang and flavor words per image
    for brand in flavor_slang_by_brand['brand']:
        brand_df = df_top_brands[df_top_brands['brand'] == brand]
        flavor_slang_by_brand.loc[flavor_slang_by_brand['brand'] == brand, 'avg_slang_per_image'] = brand_df['slang_word_count'].sum() / len(brand_df)
        flavor_slang_by_brand.loc[flavor_slang_by_brand['brand'] == brand, 'avg_flavor_per_image'] = brand_df['flavor_word_count'].sum() / len(brand_df)

    flavor_slang_by_brand = flavor_slang_by_brand.sort_values('avg_flavor_per_image', ascending=False)

    plt.figure(figsize=(10,10))
    sns.barplot(y='brand', x='avg_flavor_per_image', data=flavor_slang_by_brand, color='skyblue', label='Flavor Words')
    plt.title('Average Flavor Words per Image by Brand', fontsize=14)
    plt.xlabel('Average Flavor Words per Image')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Average_Flavor_Words_per_Image_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    plt.figure(figsize=(10,10))
    sns.barplot(y='brand', x='avg_slang_per_image', data=flavor_slang_by_brand, color='salmon', label='Slang Words')
    plt.title('Average Slang Words per Image by Brand', fontsize=14)
    plt.xlabel('Average Slang Words per Image')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Average_Slang_Words_per_Image_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    # 4. Reading Grade Level
    reading_grade_by_brand = df_top_brands.groupby('brand')['reading_grade'].mean().reset_index()
    reading_grade_by_brand = reading_grade_by_brand.sort_values('reading_grade')

    plt.figure(figsize=(10,10))
    sns.barplot(x='reading_grade', y='brand', data=reading_grade_by_brand, palette='viridis')
    plt.title('Reading Grade Level by Brand', fontsize=14)
    plt.xlabel('Average Reading Grade Level')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Reading_Grade_Level_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    # 5. Flesch Readability Score
    flesch_by_brand = df_top_brands.groupby('brand')['flesch_score'].mean().reset_index()
    flesch_by_brand = flesch_by_brand.sort_values('flesch_score', ascending=False)

    plt.figure(figsize=(10,10))
    sns.barplot(x='flesch_score', y='brand', data=flesch_by_brand, palette='viridis')
    plt.title('Flesch Readability Score by Brand (Higher = Easier to Read)', fontsize=14)
    plt.xlabel('Average Flesch Readability Score')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Flesch_Readability_Score_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    # 6. Special formatting elements
    special_format_by_brand = pd.DataFrame({
        'brand': df_top_brands['brand'].unique()
    })

    # Calculate average formatting elements per image
    for brand in special_format_by_brand['brand']:
        brand_df = df_top_brands[df_top_brands['brand'] == brand]
        special_format_by_brand.loc[special_format_by_brand['brand'] == brand, 'avg_exclamation'] = brand_df['exclamation_count'].sum() / len(brand_df)
        special_format_by_brand.loc[special_format_by_brand['brand'] == brand, 'avg_all_caps'] = brand_df['all_caps_words'].sum() / len(brand_df)
        special_format_by_brand.loc[special_format_by_brand['brand'] == brand, 'avg_hashtag'] = brand_df['hashtag_count'].sum() / len(brand_df)
        special_format_by_brand.loc[special_format_by_brand['brand'] == brand, 'avg_at_mention'] = brand_df['at_mention_count'].sum() / len(brand_df)

    # Sort by ALL CAPS usage
    special_format_by_brand = special_format_by_brand.sort_values('avg_all_caps', ascending=False)

    plt.figure(figsize=(10,10))
    sns.barplot(y='brand', x='avg_all_caps', data=special_format_by_brand, color='purple')
    plt.title('Average ALL CAPS Words per Image by Brand', fontsize=14)
    plt.xlabel('Average ALL CAPS Words per Image')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Average_ALL_CAPS_Words_per_Image_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    plt.figure(figsize=(10,10))
    sns.barplot(y='brand', x='avg_exclamation', data=special_format_by_brand, color='orange')
    plt.title('Average Exclamation Marks per Image by Brand', fontsize=14)
    plt.xlabel('Average Exclamation Marks per Image')
    plt.ylabel('Brand')
    image_path = os.path.join(visualization_folder, "Average_Exclamation_Marks_per_Image_Brand.png")
    plt.savefig(image_path)
    plt.close()
    
    # 7. Youth Appeal vs Reading Grade
    plt.figure(figsize=(10,10))
    sns.scatterplot(data=youth_appeal_by_brand.merge(reading_grade_by_brand),
                    x='reading_grade', y='youth_appeal_score',
                    size='youth_appeal_score', sizes=(50, 200),
                    hue='brand', palette='viridis')
    plt.title('Youth Appeal Score vs Reading Grade Level by Brand', fontsize=14)
    plt.xlabel('Reading Grade Level')
    plt.ylabel('Youth Appeal Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    image_path = os.path.join(visualization_folder, "Youth_Appeal_Score_vs_Reading_Grade_Level.png")
    plt.savefig(image_path)
    plt.close()

    # 8. Vape Types by Brand
    # Ensure 'Vape_Type' column exists and clean it
    if 'Vape_Type' in df.columns:
        vape_types = ['mod', 'electronic-cigarette', 'pod', 'unknown']
        vape_types_by_brand = pd.DataFrame({
            'brand': top_brands
        })

        for vape_type in vape_types:
            for brand in top_brands:
                brand_df = df[df['brand'] == brand]
                count = len(brand_df[brand_df['Vape_Type'] == vape_type])
                vape_types_by_brand.loc[vape_types_by_brand['brand'] == brand, vape_type] = count

        # Replace NaN with 0
        vape_types_by_brand = vape_types_by_brand.fillna(0)

        plt.figure(figsize=(10,10))
        vape_types_by_brand.set_index('brand')[vape_types].plot(kind='barh', stacked=True, figsize=(10, 8), colormap='viridis')
        plt.title('Vape Types by Brand', fontsize=14)
        plt.xlabel('Count')
        plt.ylabel('Brand')
        plt.legend(title='Vape Type')
        image_path = os.path.join(visualization_folder, "Vape_types_by_brand.png")
        plt.savefig(image_path)
        plt.close()    


    # Additional analysis: Correlation heatmap of text-based features
    text_features = ['slang_ratio', 'flavor_ratio', 'youth_appeal_score', 'reading_grade',
                     'flesch_score', 'avg_word_length', 'complex_word_ratio']
    if 'cool_factor_ratio' in df.columns:
        text_features.append('cool_factor_ratio')

    # Create correlation matrix
    corr = df[text_features].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Between Text-Based Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('static/visualization_images/text_features_correlation.png', dpi=300)
     

    # Brand-specific analysis using radar charts with Plotly
    def create_radar_chart(brand):
        # Get the data for the specific brand
        brand_df = df[df['brand'] == brand]

        # Calculate average values
        avg_youth_appeal = brand_df['youth_appeal_score'].mean()
        avg_reading_grade = brand_df['reading_grade'].mean() / 14  # Normalize to 0-1 scale
        avg_flesch_score = brand_df['flesch_score'].mean() / 100  # Normalize to 0-1 scale
        avg_flavor_ratio = brand_df['flavor_ratio'].mean() / 0.2  # Normalize to 0-1 scale
        avg_slang_ratio = brand_df['slang_ratio'].mean() / 0.03  # Normalize to 0-1 scale
        avg_all_caps = brand_df['all_caps_words'].mean() / 6  # Normalize to 0-1 scale

        # Create the radar chart
        categories = ['Youth Appeal', 'Reading Grade', 'Flesch Score (Inverted)',
                      'Flavor Words', 'Slang Words', 'ALL CAPS']

        values = [avg_youth_appeal, avg_reading_grade, 1 - avg_flesch_score,  # Invert Flesch so higher means harder
                  avg_flavor_ratio, avg_slang_ratio, avg_all_caps]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=brand
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"{brand} Marketing Profile Analysis",
            showlegend=False
        )
        save_path = f'static/visualization_images/{brand}/{brand}_radar.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        

    # Create radar charts for the top 5 brands by count
    for brand in brand_counts.head(5)['brand']:
        create_radar_chart(brand)
        
        
        

    # Create a comparison of all top brands in one radar chart
    def create_brand_comparison_radar():
        top5_brands = brand_counts.head(5)['brand'].tolist()

        fig = go.Figure()

        for brand in top5_brands:
            brand_df = df[df['brand'] == brand]

            # Calculate average values
            avg_youth_appeal = brand_df['youth_appeal_score'].mean()
            avg_reading_grade = brand_df['reading_grade'].mean() / 14  # Normalize to 0-1 scale
            avg_flesch_score = brand_df['flesch_score'].mean() / 100  # Normalize to 0-1 scale
            avg_flavor_ratio = brand_df['flavor_ratio'].mean() / 0.2  # Normalize to 0-1 scale
            avg_slang_ratio = brand_df['slang_ratio'].mean() / 0.03  # Normalize to 0-1 scale
            avg_all_caps = brand_df['all_caps_words'].mean() / 6  # Normalize to 0-1 scale

            # Create the radar chart
            categories = ['Youth Appeal', 'Reading Grade', 'Flesch Score (Inverted)',
                          'Flavor Words', 'Slang Words', 'ALL CAPS']

            values = [avg_youth_appeal, avg_reading_grade, 1 - avg_flesch_score,  # Invert Flesch so higher means harder
                      avg_flavor_ratio, avg_slang_ratio, avg_all_caps]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=brand
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            title="Brand Comparison: Marketing Profile Analysis",
        )
        
        save_path = f'static/visualization_images/brand_comparison_radar.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
        

    # Show the comparison radar chart
    create_brand_comparison_radar()

    # Create a summary table of key text metrics by brand
    summary_table = pd.DataFrame({
        'brand': top_brands
    })

    # Calculate metrics for each brand
    for brand in summary_table['brand']:
        brand_df = df[df['brand'] == brand]

        summary_table.loc[summary_table['brand'] == brand, 'count'] = len(brand_df)
        summary_table.loc[summary_table['brand'] == brand, 'youth_appeal_score'] = brand_df['youth_appeal_score'].mean()
        summary_table.loc[summary_table['brand'] == brand, 'reading_grade'] = brand_df['reading_grade'].mean()
        summary_table.loc[summary_table['brand'] == brand, 'flesch_score'] = brand_df['flesch_score'].mean()
        summary_table.loc[summary_table['brand'] == brand, 'avg_flavor_words'] = brand_df['flavor_word_count'].sum() / len(brand_df)
        summary_table.loc[summary_table['brand'] == brand, 'avg_slang_words'] = brand_df['slang_word_count'].sum() / len(brand_df)
        summary_table.loc[summary_table['brand'] == brand, 'avg_all_caps'] = brand_df['all_caps_words'].sum() / len(brand_df)
        summary_table.loc[summary_table['brand'] == brand, 'youth_reading_level_pct'] = (brand_df['youth_reading_level'].sum() / len(brand_df)) * 100

        # Fix for the contains_warning column - check if it's a string and handle accordingly
        if 'contains_warning' in brand_df.columns:
            if brand_df['contains_warning'].dtype == 'object':  # If it's a string
                # Count occurrences where contains_warning is 'True' (as a string)
                warning_count = brand_df['contains_warning'].str.lower().isin(['true', 'yes', '1']).sum()
            else:  # If it's boolean or numeric
                warning_count = brand_df['contains_warning'].sum()

            summary_table.loc[summary_table['brand'] == brand, 'contains_warning_pct'] = (warning_count / len(brand_df)) * 100
        else:
            summary_table.loc[summary_table['brand'] == brand, 'contains_warning_pct'] = 0

    # Display the summary table
    summary_table = summary_table.sort_values('count', ascending=False)
    summary_table = summary_table.round(3)
    print("\nSummary Table of Text Metrics by Brand:")
    print(summary_table)

    # Export the summary table to CSV
    summary_table.to_csv('vape_brand_text_metrics_summary.csv', index=False)

    # Additional advanced visualizations with Plotly
    # Youth appeal vs flavor words scatter plot
    fig = px.scatter(summary_table,
                    x='avg_flavor_words',
                    y='youth_appeal_score',
                    size='count',
                    color='brand',
                    hover_name='brand',
                    text='brand',
                    title='Youth Appeal Score vs. Flavor Words by Brand',
                    labels={'avg_flavor_words': 'Average Flavor Words per Image',
                           'youth_appeal_score': 'Youth Appeal Score',
                           'count': 'Number of Images'})

    fig.update_traces(textposition='top center')
    fig.update_layout(height=600, width=800)

    # Create a function to generate histograms for text features by brand
    def plot_feature_distribution(feature, title, top_n=5):
        plt.figure(figsize=(12, 6))

        for brand in brand_counts.head(top_n)['brand']:
            brand_data = df[df['brand'] == brand][feature].dropna()
            if len(brand_data) > 0:
                sns.kdeplot(brand_data, label=brand)

        plt.title(title, fontsize=14)
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        output_path = f'static/visualization_images/{feature}_distribution.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        
    brand_counts = df['brand'].value_counts().reset_index()
    brand_counts.columns = ['brand', 'count']
    # Plot distributions for key text features
    plot_feature_distribution('youth_appeal_score', 'Youth Appeal Score Distribution by Top 5 Brands')
    plot_feature_distribution('reading_grade', 'Reading Grade Level Distribution by Top 5 Brands')
    plot_feature_distribution('flesch_score', 'Flesch Readability Score Distribution by Top 5 Brands')


    df.columns



    # # Load the data
    # df = pd.read_csv('vape_ocr_2.csv')

    # Get a list of all columns after 'tokens'
    all_columns = df.columns.tolist()
    token_index = all_columns.index('tokens')
    columns_after_tokens = all_columns[token_index+1:]

    # Exclude flag columns and specified columns
    columns_to_exclude = [
        'contains_warning', 'text_targets_youth', 'simple_language_flag',
        'youth_reading_level', 'special_chars_youth_flag', 'contains_emojis',
        'excessive_punctuation', 'special_formatting', 'token_string', 'flesch_score'
    ]

    # Create the list of columns to include in the bar graphs
    columns_to_include = [col for col in columns_after_tokens if col not in columns_to_exclude]

    # Special columns that need different color highlighting
    highlight_columns = [
        'youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score'
    ]

    # Get a list of all brands
    brands = df['brand'].unique().tolist()

    # Function to create bar graphs for each brand
    def create_brand_bar_graph(brand_name):
        # Filter data for this brand
        brand_df = df[df['brand'] == brand_name]

        # Calculate average values for each column
        avg_values = {}
        for col in columns_to_include:
            if col in brand_df.columns:
                # Check if column has numeric data
                if pd.api.types.is_numeric_dtype(brand_df[col]):
                    avg_values[col] = brand_df[col].mean()

        # Create a DataFrame for plotting
        avg_df = pd.DataFrame({'metric': list(avg_values.keys()), 'value': list(avg_values.values())})

        # Create the bar colors list (highlight specific columns)
        bar_colors = ['#ff7f0e' if col in highlight_columns else '#1f77b4' for col in avg_df['metric']]

        # Create the plot
        plt.figure(figsize=(15, 10))
        bars = plt.bar(avg_df['metric'], avg_df['value'], color=bar_colors)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)

        plt.title(f'Average Text Metrics for {brand_name}', fontsize=16)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Average Value', fontsize=14)
        plt.xticks(rotation=90, fontsize=10)
        plt.tight_layout()

        # Create a legend for the colors

        legend_elements = [
            Patch(facecolor='#ff7f0e', label='Youth Appeal Metrics'),
            Patch(facecolor='#1f77b4', label='Other Metrics')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        output_dir = f'static/visualization_images/{brand_name}'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        plt.savefig(f'static/visualization_images/{brand_name}/{brand_name}_metrics.png', dpi=300, bbox_inches='tight')
       
        plt.close()

    # Create a summary comparison graph for major brands
    def create_comparative_analysis():
        # Get the top brands by image count
        brand_counts = df['brand'].value_counts()
        top_brands = brand_counts[brand_counts >= 50].index.tolist()

        # Create a DataFrame to store the metrics for top brands
        comparative_df = pd.DataFrame()

        # Selected important metrics for comparison
        key_metrics = [
            'youth_appeal_score', 'slang_ratio', 'flavor_ratio', 'reading_grade',
            'flesch_score', 'avg_word_length', 'readability_youth_score',
            'special_chars_youth_score'
        ]

        # Calculate average values for each metric and brand
        for brand in top_brands:
            brand_df = df[df['brand'] == brand]
            brand_metrics = {'brand': brand}

            for metric in key_metrics:
                if metric in brand_df.columns:
                    if pd.api.types.is_numeric_dtype(brand_df[metric]):
                        brand_metrics[metric] = brand_df[metric].mean()

            # Add to the comparative DataFrame
            comparative_df = pd.concat([comparative_df, pd.DataFrame([brand_metrics])], ignore_index=True)

        # Create comparison plots for key metrics
        for metric in key_metrics:
            if metric in comparative_df.columns:
                plt.figure(figsize=(12, 8))

                # Sort brands by metric value
                sorted_df = comparative_df.sort_values(by=metric)

                # Use different colors for youth appeal metrics
                bar_color = '#ff7f0e' if metric in highlight_columns else '#1f77b4'

                # Create bar chart
                bars = plt.bar(sorted_df['brand'], sorted_df[metric], color=bar_color)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', rotation=0)

                plt.title(f'Comparison of {metric} Across Top Brands', fontsize=16)
                plt.xlabel('Brand', fontsize=14)
                plt.ylabel(metric, fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                # Save the comparative plot
                plt.savefig(f'static/visualization_images/comparison_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()

    # Generate individual brand graphs
    for brand in brands:
        print(f"Generating graph for {brand}...")
        
        create_brand_bar_graph(brand)

    # Generate comparative analysis graphs
    print("Generating comparative analysis graphs...")
    create_comparative_analysis()

    print("All graphs generated successfully!")

    # Get all brands
    all_brands = df['brand'].value_counts().sort_values(ascending=False).index.tolist()

    # 1. Contains Warning Comparison
    plt.figure(figsize=(14, 10))
    warning_percentages = []
    brands_list = []

    for brand in all_brands:
        brand_df = df[df['brand'] == brand]
        if 'contains_warning' in brand_df.columns:
            # Handle different data types
            if brand_df['contains_warning'].dtype == 'object':
                # If it's a string column, count 'True', 'Yes', etc.
                warning_count = brand_df['contains_warning'].str.lower().isin(['true', 'yes', '1']).sum()
            else:
                # Otherwise assume it's boolean or numeric
                warning_count = brand_df['contains_warning'].sum()

            percentage = (warning_count / len(brand_df)) * 100
            warning_percentages.append(percentage)
            brands_list.append(brand)

    # Create DataFrame for sorting
    warning_df = pd.DataFrame({'brand': brands_list, 'warning_percentage': warning_percentages})
    warning_df = warning_df.sort_values('warning_percentage', ascending=False)

    # Create bar plot
    bars = plt.bar(warning_df['brand'], warning_df['warning_percentage'], color='#d62728')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', rotation=0)

    plt.title('Percentage of Images with Warning Text by Brand', fontsize=16)
    plt.xlabel('Brand', fontsize=14)
    plt.ylabel('Percentage with Warnings (%)', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('static/visualization_images/contains_warning_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Function to create score comparisons for the other 3 metrics
    def create_score_comparison(score_column, title, output_filename):
        if score_column not in df.columns:
            print(f"Column {score_column} not found in dataset")
            return

        # Calculate average scores
        average_scores = []
        brands_with_data = []

        for brand in all_brands:
            brand_df = df[df['brand'] == brand]
            if pd.api.types.is_numeric_dtype(brand_df[score_column]):
                avg_score = brand_df[score_column].mean()
                average_scores.append(avg_score)
                brands_with_data.append(brand)

        # Create DataFrame for sorting
        score_df = pd.DataFrame({'brand': brands_with_data, 'average_score': average_scores})
        score_df = score_df.sort_values('average_score', ascending=False)

        # Create bar plot
        plt.figure(figsize=(14, 10))
        bars = plt.bar(score_df['brand'], score_df['average_score'], color='#ff7f0e')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)

        plt.title(title, fontsize=16)
        plt.xlabel('Brand', fontsize=14)
        plt.ylabel(f'Average {score_column}', fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Youth Appeal Score comparison
    create_score_comparison(
        'youth_appeal_score',
        'Youth Appeal Score Comparison Across All Brands',
        'static/visualization_images/youth_appeal_score_comparison.png'
    )

    # 3. Readability Youth Score comparison
    create_score_comparison(
        'readability_youth_score',
        'Readability Youth Score Comparison Across All Brands',
        'static/visualization_images/readability_youth_score_comparison.png'
    )

    # 4. Special Chars Youth Score comparison
    create_score_comparison(
        'special_chars_youth_score',
        'Special Characters Youth Score Comparison Across All Brands',
        'static/visualization_images/special_chars_youth_score_comparison.png'
    )

    print("All 4 comparison graphs created successfully!")


    # Define columns to use for radar chart
    radar_columns = [
        'slang_word_count', 'flavor_word_count', 'social_media_count',
        'urgency_count', 'identity_count', 'excitement_count',
        'avg_word_length', 'complex_word_ratio', 'cool_factor_count',
        'distinctive_term_score', 'youth_appeal_score'
    ]

    # Verify columns exist in the dataset
    for col in radar_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataset")
            radar_columns.remove(col)

    # Get all brands with at least 10 images
    brand_counts = df['brand'].value_counts()
    brands_to_include = brand_counts[brand_counts >= 10].index.tolist()

    # Function to create a radar chart
    def radar_factory(num_vars, frame='circle'):
        """Create a radar chart with `num_vars` axes."""
        # Calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarAxes(PolarAxes):
            name = 'radar'

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill to handle closed polygons"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot to handle closed polygons"""
                lines = super().plot(*args, **kwargs)
                return lines

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    spine_type = 'circle'
                    spine_path = Path.unit_regular_polygon(num_vars)
                    spine_pathT = Affine2D().scale(.5).translate(.5, .5).transform_path(spine_path)
                    spine = Spine(self, spine_type, spine_pathT)
                    return {'polar': spine}
                else:
                    raise ValueError("Unknown value for 'frame': %s" % frame)

        # Register the custom axes class
        register_projection(RadarAxes)
        return theta

    # Function to create normalized radar data for a brand
    def create_radar_data(brand_name):
        brand_df = df[df['brand'] == brand_name]

        # Calculate sums for each column
        data = []
        for col in radar_columns:
            if pd.api.types.is_numeric_dtype(brand_df[col]):
                # Sum the values for count columns, average for ratio columns
                if 'count' in col or col in ['distinctive_term_score', 'youth_appeal_score']:
                    val = brand_df[col].sum()
                else:
                    val = brand_df[col].mean()
                data.append(val)
            else:
                data.append(0)

        return data

    # Create a comparative DataFrame for normalization
    comparative_df = pd.DataFrame(columns=radar_columns)

    for brand in brands_to_include:
        brand_data = create_radar_data(brand)
        comparative_df.loc[brand] = brand_data

    # Normalize data for better visualization
    # For each column, calculate max value for scaling
    max_values = {}
    for col in radar_columns:
        max_values[col] = comparative_df[col].max()

    # Function to generate radar chart for a brand
    def generate_brand_radar(brand):
        # Create a directory to store the radar charts
        data = comparative_df.loc[brand].values

        # Normalize data (0-1 scale)
        normalized_data = []
        for i, val in enumerate(data):
            if max_values[radar_columns[i]] > 0:
                normalized_data.append(val / max_values[radar_columns[i]])
            else:
                normalized_data.append(0)

        # Create the radar chart
        N = len(radar_columns)
        theta = radar_factory(N, frame='polygon')

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

        # Plot the data and fill the area
        ax.plot(theta, normalized_data, 'o-', linewidth=2)
        ax.fill(theta, normalized_data, alpha=0.25)

        # Set labels and title
        ax.set_varlabels(radar_columns)
        plt.title(f'Text Features Profile: {brand}', size=15, y=1.1)

        # Add original values as annotations
        for i, value in enumerate(data):
            angle = theta[i]
            x = 0.5 + normalized_data[i] * 0.5 * np.cos(angle)
            y = 0.5 + normalized_data[i] * 0.5 * np.sin(angle)
            plt.annotate(f'{value:.2f}', xy=(x, y), xytext=(x, y),
                         fontsize=8, ha='center', va='center')

        # Save the chart
        plt.tight_layout()
        plt.savefig(f'static/visualization_images/{brand}/{brand}_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

        return normalized_data

    # Generate individual radar charts for each brand
    for brand in brands_to_include:
        print(f"Generating radar chart for {brand}...")
        generate_brand_radar(brand)

    # Additionally, create a comparative radar chart with the top 5 brands
    plt.figure(figsize=(12, 10))
    top_5_brands = brand_counts.head(5).index.tolist()

    # Create the radar chart
    N = len(radar_columns)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='radar'))

    # Plot data for each brand with different colors
    colors = ['b', 'r', 'g', 'c', 'm']
    for i, brand in enumerate(top_5_brands):
        data = comparative_df.loc[brand].values

        # Normalize data
        normalized_data = []
        for j, val in enumerate(data):
            if max_values[radar_columns[j]] > 0:
                normalized_data.append(val / max_values[radar_columns[j]])
            else:
                normalized_data.append(0)

        ax.plot(theta, normalized_data, 'o-', linewidth=2, color=colors[i], label=brand)
        ax.fill(theta, normalized_data, alpha=0.1, color=colors[i])

    # Set labels, title and legend
    ax.set_varlabels(radar_columns)
    plt.title('Text Features Profile: Top 5 Brands Comparison', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig('static/visualization_images/top_5_brands_comparison_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All radar charts generated successfully!")

    # Create a summary table of the radar metrics
    summary_df = comparative_df.copy()
    summary_df['total_images'] = brand_counts[summary_df.index].values

    # Add derived metrics for easier interpretation
    summary_df['flavor_to_slang_ratio'] = summary_df['flavor_word_count'] / summary_df['slang_word_count'].replace(0, 0.001)
    summary_df['social_engagement_ratio'] = summary_df['social_media_count'] / summary_df.index.map(lambda x: len(df[df['brand'] == x]))

    # Round to 2 decimal places for readability
    summary_rounded = summary_df.round(2)

    # Save the summary table
 
    csv_path = 'static/brand_text_features_summary.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    summary_rounded.to_csv(csv_path)

    print("Summary table of text features saved to CSV.")

    