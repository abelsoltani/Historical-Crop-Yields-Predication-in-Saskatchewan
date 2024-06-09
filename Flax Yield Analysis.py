#!/usr/bin/env python
# coding: utf-8

# # Project Overview
# 
# ## Problem Being Solved
# The agricultural industry in Saskatchewan is a significant contributor to the economy, and optimizing crop production is essential for sustaining growth and profitability. This project aims to address these challenges by utilizing rural manucipality crop yeilds and Geographic Information System (GIS) datas, and machine learning techniques to identify the optimal rural municipality in Saskatchewan for flax investment and to propose suitable crop rotation plans.
# 
# ## Objective of the Project
# The primary objective of this project is to leverage GIS data and machine learning models to provide actionable insights for farmers and investors in Saskatchewan. Specifically, the project aims to:
# 
# 1. **Identify the Best Rural Municipality for Flax Investment**: Using crop yeilds and GIS datas to analyze historical crop yields, the project seeks to pinpoint the rural municipality in Saskatchewan that offers the most favorable farms for flax cultivation.
# 
# 2. **Determine Optimal Crop Rotation Strategies**: To ensure sustainable agricultural practices and enhance soil health, the project will explore different crop rotation options that can be integrated with flax farming.
# 
# ## Questions Being Answered
# To achieve the above objectives, the project will answer the following key questions:
# 
# 1. **Where is the Best Rural Municipality in Saskatchewan to Invest in Flax?**
# 
# 2. **What Crop Rotation Can We Consider for Flax Farming?**
# 
# 
# ## Methodology
# To answer these questions, the project employs two machine learning models: unsupervised spectral clustering and K-means clustering.
# 
# 1. **Unsupervised Spectral Clustering**
# 
# 2. **K-means Clustering**
# 
# By combining these two clustering techniques, the project provides a comprehensive analysis of the rural municipalities in Saskatchewan, offering a clear recommendations for flax investment and sustainable crop rotation practices.
# 

# In[1]:


# Importing Libraries

import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations and matrix calculations
import geopandas as gpd  # Extends Pandas to allow spatial operations on geometric types
import seaborn as sb  # Statistical data visualization based on matplotlib
import matplotlib.pyplot as plt  # 2D plotting library for creating static, animated, and interactive visualizations
import ipywidgets as widgets  # Interactive HTML widgets for Jupyter notebooks and IPython
from IPython.display import display  # Display rich content in Jupyter notebooks
from ipywidgets import interact  # Creates interactive user interface controls in notebooks
from sklearn.preprocessing import StandardScaler  # Standardize features by removing the mean and scaling to unit variance
from sklearn.cluster import SpectralClustering  # Spectral clustering for partitioning data into clusters
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.metrics import silhouette_score  # Evaluates clustering performance using silhouette analysis


# ### Loading Crop Yields and GIS Data for Rural Municipalities
# 

# In[2]:


import os  # Import the os module for path manipulation

# Define the main directory path
main_path = '/Users/abels/Desktop/Pallet Skills/Courses/Stream 3/'

# Concatenate the main directory path with the filenames to get the full file paths
df_rm_yields_path = os.path.join(main_path, 'rm-yields-data.csv')
gdf_rm_path = os.path.join(main_path + 'RM_shapefile', 'RuralMunicipality.shp')

# # Reading 2000-2023 Aggregated Yield Data
df_rm_yields = pd.read_csv(df_rm_yields_path)
gdf_rm = gpd.read_file(gdf_rm_path)


# In[3]:


df_rm_yields.info()


# In[4]:


df_rm_yields.head()


# In[5]:


df_rm_yields.columns


# In[7]:


# Plot the GeoDataFrame with customizations
fig, ax = plt.subplots(figsize=(10, 8))

gdf_rm.plot(ax=ax, 
         cmap='Greens',  # Use a color map (e.g., 'viridis')
         legend=False,     # Show legend
         legend_kwds={'label': "Legend Title", 'orientation': "horizontal"},  # Legend settings
         linewidth=0.5,    # Adjust line width
         edgecolor='gray', # Adjust edge color
         alpha=0.7        # Adjust transparency
        )

# Add title
plt.title('Saskatchewan Rural Municipalities', fontsize=10)
plt.xticks(rotation=45)
# Add scale bar
#scalebar = gpd.plotting.add_scalebar(ax, location='lower right', length=100, linewidth=2, color='black', units='km')

# Add gridlines
ax.grid(True, linestyle='--', linewidth=0.5)

# Add labels to axes
#ax.set_xlabel('Longitude', fontsize=12)
#ax.set_ylabel('Latitude', fontsize=12)

# Add annotations
ax.annotate('Annotation Text', xy=(0.6, 0.6), xytext=(0.6, 0.6),
            arrowprops=dict(facecolor='black', shrink=0.05))

# Show the plot
plt.show()


# In[8]:


df_major_crops=df_rm_yields[['Year', 'RM', 'Canola', 'Spring Wheat',
       'Durum','Oats', 'Lentils', 'Peas', 'Barley', 'Flax']]
# Changing Pounds to bushels
df_major_crops.loc[:, 'Lentils'] = df_major_crops['Lentils'] / 60
df_major_crops.describe().T


# In[9]:


gdf_rm.head()


# In[10]:


gdf_rm_clean= gdf_rm[['RMNO', 'RMNM', 'geometry']]
gdf_rm_clean.head()


# In[11]:


print(gdf_rm_clean.info())
print("%" * 40)
print(df_major_crops.info())


# In[12]:


# Type changing
gdf_rm_clean.loc[:, 'RMNO'] = gdf_rm_clean['RMNO'].astype(int)

# Merging Yield data with GIS
gdf_rm_yield= pd.merge(gdf_rm_clean.rename(columns={'RMNO':'RM'}), df_major_crops, on='RM', how='inner')
gdf_rm_yield


# ## Exploratory Data Analysis

# In[13]:


import ipywidgets as widgets
import matplotlib.pyplot as plt

# List of crops to include in plots
crops = ['Flax', 'Canola', 'Spring Wheat', 'Durum', 'Oats', 'Lentils', 'Peas', 'Barley', ]

# Function to plot yield data for a specific crop and year range
def plot_yield_by_year(crop, year_range):
    # Filter the years based on the selected range
    years = list(range(year_range[0], year_range[1] + 1))
    
    # Calculate the number of rows and columns needed for the subplots
    num_plots = len(years)
    rows = (num_plots // 3) + (1 if num_plots % 3 else 0)
    cols = min(num_plots, 3)
    
    # Set up the figure with the appropriate number of rows and columns
    fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 5))
    fig.suptitle(f'{crop} Yield per Year ({years[0]} - {years[-1]})', color='black', size=20)
    
    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Loop through each year and plot it on its respective subplot
    for i, year in enumerate(years):
        ax = axs[i]
        gdf_rm_yield[gdf_rm_yield['Year'] == year].plot(
            column=crop,
            cmap='Purples',
            legend=True,
            ax=ax,
            edgecolor='black'
        )
        ax.set_title(f'Year: {year}', color='black', size=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove unused subplots if there are any
    for j in range(len(years), len(axs)):
        fig.delaxes(axs[j])
    
    # Adjust the spacing between subplots for readability
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Create a dropdown widget for selecting crops
crop_slider = widgets.Dropdown(
    options=crops,
    value= 'Flax', # Default to selecting 'Falx' -- value= for selecting all crops ,
    description='Select Crop:',
    disabled=False,
)

# Create a range slider widget for selecting years
year_slider = widgets.IntRangeSlider(
    value=[2015, 2023],
    min=2000,
    max=2023,
    step=1,
    description='Select Years:',
    continuous_update=False,
)

# Function to update the plot based on the selected crop and year range
def update_plot(crop, year_range):
    plot_yield_by_year(crop, year_range)

# Display the sliders and link them to the update function
widgets.interact(update_plot, crop=crop_slider, year_range=year_slider)


# In[14]:


# crops - is a list defined in mapping cell
# >0.2 slight correlation
# >0.4 Moderate corrleation
# > 0.6 High
# > 0.8 Very correlation 

# Pearson Correlation
sb.heatmap(df_major_crops.loc[df_major_crops['Year']>2000][crops].corr(),
           annot=True,
           cmap='RdYlGn')

# Rank correlatation


# In[26]:


# Function to plot the data for a given crop

def plot_data(selected_crop):
    merged_df.plot(selected_crop, cmap='Purples', legend=True)
    plt.title(f'Historical Average | {selected_crop}')
    plt.show()

# Create the crop selection widget
crop_selection = widgets.Dropdown(
    options=crops,
    value= 'Flax',  # Default to 'Flax'
    description='Crop',
    disabled=False
)

# Use interact to create the interactive plot
interact(plot_data, selected_crop=crop_selection)


# ## Outliers
# ### Before treating

# In[15]:


from ipywidgets import interact

filtered_df = df_major_crops[(df_major_crops['Year'] >= 2000) & (df_major_crops['Year'] <= 2023)]

# Function to plot the boxplots for a given year range
def plot_boxplots(year_range, selected_crops):
    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 16))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Create a list of years for the range
    years = list(range(year_range[0], year_range[1] + 1))  # Display years within the selected range
    
    # Check if the selected crops exist in the DataFrame
    existing_crops = [crop for crop in selected_crops if crop in filtered_df.columns]
    
    # Iterate through the years and create a boxplot for each crop
    for i, year in enumerate(years):
        if i < len(axes):
            ax = axes[i]
            year_data = filtered_df[filtered_df['Year'] == year]
            year_data.boxplot(column=existing_crops, ax=ax)
            ax.set_title(f'Year: {year}', size=12, color='teal')
            ax.tick_params(axis='x', rotation=30)  # Rotate x-tick labels
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

# Create the year range slider
year_range_slider = widgets.IntRangeSlider(
    value=[2000, 2024], 
    min=2000, 
    max=2023, 
    step=1, 
    description='Year Range'
)

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value= crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Use interact to create the interactive plot
interact(plot_boxplots, year_range=year_range_slider, selected_crops=crop_selection)


# ## After treating

# In[16]:


# Calculate mean and standard deviation for each crop
means = df_major_crops[crops].mean()
stds = df_major_crops[crops].std()

# Determine the clipping bounds
lower_bounds = means - 3 * stds
upper_bounds = means + 3 * stds

# Clip the data
df_clipped = df_major_crops.copy()
for crop in crops:
    df_clipped[crop] = df_major_crops[crop].clip(lower=lower_bounds[crop], upper=upper_bounds[crop])

# Function to plot the boxplots for a given year range
def plot_boxplots(year_range, selected_crops):
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(20, 16))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Create a list of years for the range
    years = list(range(year_range[0], year_range[1] + 1))  # Display years within the selected range

    # Check if the selected crops exist in the DataFrame
    existing_crops = [crop for crop in selected_crops if crop in df_clipped.columns]

    # Iterate through the years and create a boxplot for each crop
    for i, year in enumerate(years):
        if i < len(axes):
            ax = axes[i]
            year_data = df_clipped[df_clipped['Year'] == year]
            year_data.boxplot(column=existing_crops, ax=ax)
            ax.set_title(f'Year: {year}', size=12, color='teal')
            ax.tick_params(axis='x', rotation=30)  # Rotate x-tick labels

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Create the year range slider
year_range_slider = widgets.IntRangeSlider(
    value=[2000, 2024], 
    min=2000, 
    max=2023, 
    step=1, 
    description='Year Range'
)

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value= crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Use interact to create the interactive plot
interact(plot_boxplots, year_range=year_range_slider, selected_crops=crop_selection)


# ## Feature Construction and Selection
# 

# In[20]:


#Filtering df by year
df_00_23 = df_major_crops[df_major_crops['Year']>=2000]

# Feature Construction
df_00_23.drop(columns='Year').groupby('RM').mean()


# In[27]:


# Merging gdf_rm_clean and  df_00_23.groupby('RM').mean()

merged_df = pd.merge(gdf_rm_clean.rename(columns={'RMNO': 'RM'}), df_00_23.groupby('RM').mean(), on='RM')
merged_df


# In[28]:


# Group by 'RM' and calculate mean and standard deviation for each crop
df_agg_00_23 = df_00_23.groupby('RM')[crops].agg(['mean', 'std'])

# Flatten the column multi-index
df_agg_00_23.columns = ['_'.join(col).strip() for col in df_agg_00_23.columns.values]

# Reset index to make 'RM' a column again
df_agg_00_23.reset_index(inplace=True)


# In[29]:


df_agg_00_23


# In[30]:


#saving df to a csv file
df_agg_00_23.to_csv('/Users/abels/Desktop/Pallet Skills/Courses/Stream 3/Final Project/rm_yield_00_23-crops.csv')


# ## Methodology
# The project employs two machine learning models: unsupervised spectral clustering and K-means clustering.
# - **Spectral Clustering** is better for capturing complex, non-convex cluster shapes and handling overlapping clusters but is computationally expensive and requires a similarity matrix.
# - **K-Means Clustering** is simple, efficient, and scalable but assumes spherical clusters of similar size and struggles with non-convex clusters and varying densities.

# # Optimal Spectral Clustering Results

# In[32]:


# Function to prepare data for each crop
def prepare_data_for_crop(df, crop):
    columns = [f'{crop}_mean', f'{crop}_std']
    crop_data = df[columns].dropna().values
    return crop_data

# Standardize the data
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Function to perform spectral clustering and choose the optimal number of clusters
def spectral_clustering(data, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    labels = clustering.fit_predict(data)
    return labels

# Function to find the optimal number of clusters
def find_optimal_clusters(data, max_k):
    scores = []
    for k in range(2, max_k+1):
        labels = spectral_clustering(data, k)
        score = silhouette_score(data, labels)
        scores.append(score)
    optimal_k = scores.index(max(scores)) + 2
    return optimal_k, scores

# Function to perform clustering for selected crops
def perform_clustering(selected_crops):
    for crop in selected_crops:
        print(f"Clustering for {crop}")
        cluster_crop(crop)

# Helper function to perform clustering and visualization for a single crop
def cluster_crop(crop):
    # Prepare the data for the crop
    crop_data = prepare_data_for_crop(df_agg_00_23, crop)
    
    # Standardize the data
    crop_data_scaled = standardize_data(crop_data)
    
    # Find the optimal number of clusters
    optimal_k, scores = find_optimal_clusters(crop_data_scaled, 10)
    
    # Perform spectral clustering with the optimal number of clusters
    labels = spectral_clustering(crop_data_scaled, optimal_k)
    
    # Add the cluster labels to the original dataframe
    df_agg_00_23[f'{crop}_Spectral_Cluster_Optimal'] = np.nan
    df_agg_00_23.loc[~df_agg_00_23[[f'{crop}_mean', f'{crop}_std']].isna().any(axis=1), f'{crop}_Spectral_Cluster_Optimal'] = labels
    
    # Visualize the silhouette scores
    plt.figure()
    plt.plot(range(2, 11), scores, marker='o')
    plt.title(f'Silhouette Scores for {crop}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()
    
    # Print the results
    print(f'Optimal number of clusters for {crop}: {optimal_k}')
    print(f'Silhouette scores for {crop}: {scores}')
    
    # Visualize the clustering results
    plt.figure()
    plt.scatter(df_agg_00_23[f'{crop}_mean'], df_agg_00_23[f'{crop}_std'], c=df_agg_00_23[f'{crop}_Spectral_Cluster_Optimal'], cmap='viridis')
    plt.title(f'Optimal Spectral Clustering Results for {crop}')
    plt.xlabel(f'{crop}_mean')
    plt.ylabel(f'{crop}_std')
    plt.colorbar(label='Cluster')
    plt.show()

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value=crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Use interact to create the interactive plot
interact(perform_clustering, selected_crops=crop_selection)


# # Custom Spectral Clustering Results

# In[33]:


# crops = ['Flax', 'Canola', 'Spring Wheat', 'Durum', 'Oats', 'Lentils', 'Peas', 'Barley']

# Function to prepare data for each crop
def prepare_data_for_crop(df, crop):
    columns = [f'{crop}_mean', f'{crop}_std']
    crop_data = df[columns].dropna().values
    indices = df[columns].dropna().index
    return crop_data, indices

# Standardize the data
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Function to perform spectral clustering
def perform_spectral_clustering(data, n_clusters):
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    labels = clustering.fit_predict(data)
    return labels

# Function to perform clustering for selected crops
def perform_clustering(selected_crops, n_clusters):
    for crop in selected_crops:
        print(f"Clustering for {crop}")
        cluster_crop(crop, n_clusters)

# Helper function to perform clustering and visualization for a single crop
def cluster_crop(crop, n_clusters):
    # Prepare the data for the crop
    crop_data, indices = prepare_data_for_crop(df_agg_00_23, crop)
    
    # Standardize the data
    crop_data_scaled = standardize_data(crop_data)
    
    # Perform spectral clustering with the selected number of clusters
    labels = perform_spectral_clustering(crop_data_scaled, n_clusters)
    
    # Add the cluster labels to the original dataframe
    df_agg_00_23[f'{crop}_Spectral_Cluster_Custom'] = np.nan
    df_agg_00_23.loc[indices, f'{crop}_Spectral_Cluster_Custom'] = labels
    
    # Visualize the clustering results
    plt.figure()
    plt.scatter(df_agg_00_23[f'{crop}_mean'], df_agg_00_23[f'{crop}_std'], c=df_agg_00_23[f'{crop}_Spectral_Cluster_Custom'], cmap='viridis')
    plt.title(f'Spectral Clustering Results for {crop}')
    plt.xlabel(f'{crop}_mean')
    plt.ylabel(f'{crop}_std')
    plt.colorbar(label='Cluster')
    plt.show()

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value=crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Create the number of clusters slider
n_clusters_slider = widgets.IntSlider(
    value=5, 
    min=2, 
    max=10, 
    step=1, 
    description='Clusters'
)

# Use interact to create the interactive plot
interact(perform_clustering, selected_crops=crop_selection, n_clusters=n_clusters_slider)


# # Optimal K-mean Clustering Results
# 

# In[42]:


crops = ['Flax', 'Canola', 'Spring Wheat', 'Durum', 'Oats', 'Lentils', 'Peas', 'Barley']

# Function to prepare data for each crop
def prepare_data_for_crop(df, crop):
    columns = [f'{crop}_mean', f'{crop}_std']
    crop_data = df[columns].dropna().values
    indices = df[columns].dropna().index
    return crop_data, indices

# Standardize the data
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Function to perform KMeans clustering and get inertia
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(data)
    return labels, kmeans.inertia_

# Function to find the optimal number of clusters using the elbow method
def find_optimal_clusters(data, max_k):
    inertias = []
    for k in range(1, max_k+1):
        _, inertia = kmeans_clustering(data, k)
        inertias.append(inertia)
    
    # Elbow method to find the optimal k
    optimal_k = np.argmax(np.diff(inertias, 2)) + 2
    return optimal_k, inertias

# Function to perform clustering for selected crops
def perform_clustering(selected_crops):
    for crop in selected_crops:
        print(f"Clustering for {crop}")
        cluster_crop(crop)

# Helper function to perform clustering and visualization for a single crop
def cluster_crop(crop):
    # Prepare the data for the crop
    crop_data, indices = prepare_data_for_crop(df_agg_00_23, crop)
    
    # Standardize the data
    crop_data_scaled = standardize_data(crop_data)
    
    # Find the optimal number of clusters
    optimal_k, inertias = find_optimal_clusters(crop_data_scaled, 10)
    
    # Perform KMeans clustering with the optimal number of clusters
    labels, _ = kmeans_clustering(crop_data_scaled, optimal_k)
    
    # Add the cluster labels to the original dataframe
    df_agg_00_23[f'{crop}_KMeans_Cluster_Optimal'] = np.nan
    df_agg_00_23.loc[indices, f'{crop}_KMeans_Cluster_Optimal'] = labels
    
    # Visualize the inertia values
    plt.figure()
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title(f'Elbow Method for {crop}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.axvline(optimal_k, color='red', linestyle='--')
    plt.show()
    
    # Print the results
    print(f'Optimal number of clusters for {crop}: {optimal_k}')
    print(f'Inertia values for {crop}: {inertias}')
    
    # Visualize the clustering results
    plt.figure()
    plt.scatter(df_agg_00_23[f'{crop}_mean'], df_agg_00_23[f'{crop}_std'], c=df_agg_00_23[f'{crop}_KMeans_Cluster_Optimal'], cmap='viridis')
    plt.title(f'Optimal KMeans Clustering Results for {crop}')
    plt.xlabel(f'{crop}_mean')
    plt.ylabel(f'{crop}_std')
    plt.colorbar(label='Cluster')
    plt.show()

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value=crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Use interact to create the interactive plot
interact(perform_clustering, selected_crops=crop_selection)


# # Custom K-mean Clustering Results
# 

# In[43]:


# Function to prepare data for each crop
def prepare_data_for_crop(df, crop):
    columns = [f'{crop}_mean', f'{crop}_std']
    crop_data = df[columns].dropna().values
    return crop_data, df[columns].dropna().index

# Standardize the data
def standardize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Perform KMeans clustering with a fixed number of clusters
def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(data)
    return labels

# Function to perform clustering for selected crops and number of clusters
def perform_clustering(selected_crops, n_clusters):
    for crop in selected_crops:
        print(f"Clustering for {crop}")
        cluster_crop(crop, n_clusters)

# Helper function to perform clustering and visualization for a single crop
def cluster_crop(crop, n_clusters):
    # Prepare the data for the crop
    crop_data, indices = prepare_data_for_crop(df_agg_00_23, crop)
    
    # Standardize the data
    crop_data_scaled = standardize_data(crop_data)
    
    # Perform KMeans clustering with the selected number of clusters
    labels = perform_kmeans_clustering(crop_data_scaled, n_clusters)
    
    # Add the cluster labels to the original dataframe
    df_agg_00_23[f'{crop}_KMeans_Cluster_Custom'] = np.nan
    df_agg_00_23.loc[indices, f'{crop}_KMeans_Cluster_Custom'] = labels
    
    # Visualize the clustering results
    plt.figure()
    plt.scatter(df_agg_00_23[f'{crop}_mean'], df_agg_00_23[f'{crop}_std'], c=df_agg_00_23[f'{crop}_KMeans_Cluster_Custom'], cmap='viridis')
    plt.title(f'KMeans Clustering Results for {crop}')
    plt.xlabel(f'{crop}_mean')
    plt.ylabel(f'{crop}_std')
    plt.colorbar(label='Cluster')
    plt.show()

# Create the crop selection widget
crop_selection = widgets.SelectMultiple(
    options=crops,
    value=crops,  # Default to selecting all crops
    description='Crops',
    disabled=False
)

# Create the number of clusters slider
n_clusters_slider = widgets.IntSlider(
    value=5, 
    min=2, 
    max=10, 
    step=1, 
    description='Clusters'
)

# Use interact to create the interactive plot
interact(perform_clustering, selected_crops=crop_selection, n_clusters=n_clusters_slider)


# ## GIS Analysis to visualize areas with highest production

# In[36]:


df_agg_00_23


# In[37]:


df_agg_00_23.columns


# In[39]:


# List of crops and corresponding cluster columns
crops_clusters = {
    'Canola': ['Canola_Spectral_Cluster_Optimal', 'Canola_Spectral_Cluster_Custom', 'Canola_KMeans_Cluster_Optimal', 'Canola_KMeans_Cluster_Custom'],
    'Spring Wheat': ['Spring Wheat_Spectral_Cluster_Optimal', 'Spring Wheat_Spectral_Cluster_Custom', 'Spring Wheat_KMeans_Cluster_Optimal', 'Spring Wheat_KMeans_Cluster_Custom'],
    'Durum': ['Durum_Spectral_Cluster_Optimal', 'Durum_Spectral_Cluster_Custom', 'Durum_KMeans_Cluster_Optimal', 'Durum_KMeans_Cluster_Custom'],
    'Oats': ['Oats_Spectral_Cluster_Optimal', 'Oats_Spectral_Cluster_Custom', 'Oats_KMeans_Cluster_Optimal', 'Oats_KMeans_Cluster_Custom'],
    'Lentils': ['Lentils_Spectral_Cluster_Optimal', 'Lentils_Spectral_Cluster_Custom', 'Lentils_KMeans_Cluster_Optimal', 'Lentils_KMeans_Cluster_Custom'],
    'Peas': ['Peas_Spectral_Cluster_Optimal', 'Peas_Spectral_Cluster_Custom', 'Peas_KMeans_Cluster_Optimal', 'Peas_KMeans_Cluster_Custom'],
    'Barley': ['Barley_Spectral_Cluster_Optimal', 'Barley_Spectral_Cluster_Custom', 'Barley_KMeans_Cluster_Optimal', 'Barley_KMeans_Cluster_Custom'],
    'Flax': ['Flax_Spectral_Cluster_Optimal','Flax_Spectral_Cluster_Custom', 'Flax_KMeans_Cluster_Optimal', 'Flax_KMeans_Cluster_Custom'],
}

# Initialize a new DataFrame for ranked columns
df_agg_00_23_ranked = df_agg_00_23.copy()

# Rank the clusters based on the mean crop yield for each crop
for crop, clusters in crops_clusters.items():
    mean_column = f'{crop}_mean'
    
    for cluster_col in clusters:
        # Calculate the mean crop yield grouped by the cluster column
        cluster_means = df_agg_00_23.groupby(cluster_col).mean()[mean_column]
        
        # Rank the clusters based on the mean crop yield
        df_agg_00_23_ranked[f'{cluster_col}_ranked'] = df_agg_00_23[cluster_col].map(cluster_means.rank(method='min'))

# Drop old unranked cluster columns
for clusters in crops_clusters.values():
    df_agg_00_23_ranked.drop(columns=clusters, inplace=True)

# Function to visualize the ranking
def visualize_ranking(selected_crop):
    ranked_columns = [col for col in df_agg_00_23_ranked.columns if selected_crop in col and 'ranked' in col]
    
    for col in ranked_columns:
        plt.figure()
        plt.scatter(df_agg_00_23_ranked[f'{selected_crop}_mean'], df_agg_00_23_ranked[f'{selected_crop}_std'], c=df_agg_00_23_ranked[col], cmap='viridis')
        plt.title(f'Cluster Ranking for {selected_crop} - {col}')
        plt.xlabel(f'{selected_crop}_mean')
        plt.ylabel(f'{selected_crop}_std')
        plt.colorbar(label='Cluster Ranking')
        plt.show()

# Create the crop selection widget
crop_selection = widgets.Dropdown(
    options=crops_clusters.keys(),
    value='Flax',  # Default to 'Canola'
    description='Crop',
    disabled=False
)

# Use interact to create the interactive plot
interact(visualize_ranking, selected_crop=crop_selection)


# In[33]:


# List of crops and corresponding cluster columns
crops_clusters = {
    'Canola': ['Canola_Spectral_Cluster_Optimal', 'Canola_Spectral_Cluster_Custom', 'Canola_KMeans_Cluster_Optimal', 'Canola_KMeans_Cluster_Custom'],
    'Spring Wheat': ['Spring Wheat_Spectral_Cluster_Optimal', 'Spring Wheat_Spectral_Cluster_Custom', 'Spring Wheat_KMeans_Cluster_Optimal', 'Spring Wheat_KMeans_Cluster_Custom'],
    'Durum': ['Durum_Spectral_Cluster_Optimal', 'Durum_Spectral_Cluster_Custom', 'Durum_KMeans_Cluster_Optimal', 'Durum_KMeans_Cluster_Custom'],
    'Oats': ['Oats_Spectral_Cluster_Optimal', 'Oats_Spectral_Cluster_Custom', 'Oats_KMeans_Cluster_Optimal', 'Oats_KMeans_Cluster_Custom'],
    'Lentils': ['Lentils_Spectral_Cluster_Optimal', 'Lentils_Spectral_Cluster_Custom', 'Lentils_KMeans_Cluster_Optimal', 'Lentils_KMeans_Cluster_Custom'],
    'Peas': ['Peas_Spectral_Cluster_Optimal', 'Peas_Spectral_Cluster_Custom', 'Peas_KMeans_Cluster_Optimal', 'Peas_KMeans_Cluster_Custom'],
    'Barley': ['Barley_Spectral_Cluster_Optimal', 'Barley_Spectral_Cluster_Custom', 'Barley_KMeans_Cluster_Optimal', 'Barley_KMeans_Cluster_Custom'],
    'Flax': ['Flax_Spectral_Cluster_Optimal','Flax_Spectral_Cluster_Custom', 'Flax_KMeans_Cluster_Optimal', 'Flax_KMeans_Cluster_Custom'],
}


# Initialize a new DataFrame for ranked columns
df_agg_00_23_ranked = df_agg_00_23.copy()

# Rank the clusters based on the mean crop yield for each crop
for crop, clusters in crops_clusters.items():
    mean_column = f'{crop}_mean'
    
    for cluster_col in clusters:
        # Calculate the mean crop yield grouped by the cluster column
        cluster_means = df_agg_00_23.groupby(cluster_col).mean()[mean_column]
        
        # Rank the clusters based on the mean crop yield
        df_agg_00_23_ranked[f'{cluster_col}_ranked'] = df_agg_00_23[cluster_col].map(cluster_means.rank(method='min'))

# Drop old unranked cluster columns
for clusters in crops_clusters.values():
    df_agg_00_23_ranked.drop(columns=clusters, inplace=True)


# In[34]:


df_agg_00_23_ranked


# In[35]:


gdf_rm_clean


# In[36]:


# Changing data type 
gdf_rm['RMNO']= gdf_rm['RMNO'].astype(int)
gdf_rm_clean= gdf_rm[['RMNO', 'geometry']].rename(columns={'RMNO': 'RM'})


# In[37]:


final_df = pd.merge(gdf_rm_clean, df_agg_00_23_ranked, on = 'RM')


# In[38]:


final_df


# In[39]:


final_df.columns


# In[46]:


gpd.GeoDataFrame(final_df).explore('Flax_Spectral_Cluster_Optimal_ranked', cmap ="Purples")


# In[45]:


gpd.GeoDataFrame(final_df).explore('Flax_KMeans_Cluster_Optimal_ranked', cmap ="Purples")


# In[42]:


final_df.to_file('/Users/abels/Desktop/Pallet Skills/Courses/Stream 3/Final Project/crop_clustered_ranked_00_23.geojson', driver = 'GeoJSON')


# In[43]:


final_df.to_csv('/Users/abels/Desktop/Pallet Skills/Courses/Stream 3/Final Project/crop_clustered_ranked_00_23.csv')


# In[ ]:




