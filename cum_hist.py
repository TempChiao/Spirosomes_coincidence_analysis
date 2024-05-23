#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:21:54 2024

@author: Mathew
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Specify the path to your CSV file
csv_file_path = '/Users/Mathew/Documents/Current analysis/Spirosomes/1.csv'

# Colour map:
    
colormap = plt.cm.get_cmap('hot')

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# df = df[(df != 0).all(axis=1)]



# Plot cumulative histograms for each column and normalize
fig, ax = plt.subplots(figsize=(8, 6))


num_cols = len(df.columns)
# Plot cumulative histograms for each column as lines without filling
for i,col in enumerate(df.columns):
    color = colormap(i / num_cols)
    ax.hist(df[col], bins=200, range=[1,100000], cumulative=True, density=True, alpha=0.7, label=col, fill=False,histtype='step',color=color)

# Set labels and legend

ax.set_xlabel('Total intensity of spirosome (ADUs)')
ax.set_ylabel('Cumulative Probability')
ax.legend()
plt.savefig(csv_file_path[:-4]+"_Cumulative_Histograms.pdf")
plt.show()

df = df.replace(0, np.nan)

# Calculate mean and median of each column
column_means = df.mean()
column_medians = df.median()
column_counts = df.count()

# Create a new DataFrame with mean, median, and count values for each column
summary_df = pd.DataFrame({'Mean': column_means, 'Median': column_medians, 'Count': column_counts})


# Display the summary DataFrame
print(summary_df)

summary_df.to_csv(csv_file_path[:-4]+'_summary_statistics.csv')




fig, axs = plt.subplots(7, 2, figsize=(12, 18))

# Flatten the axs array to iterate over each subplot
axs = axs.flatten()

# Plot histograms for each column
for i, col in enumerate(df.columns):
    # Plot histogram for current column
    axs[i].hist(df[col].dropna(), bins=50, range=[1,100000], alpha=0.7, color='skyblue', edgecolor='black')
    axs[i].set_title(col)
    axs[i].set_xlabel('Intensity of spirosome',fontname='Arial', fontsize=12)
    axs[i].set_ylabel('Number of spirosomes',fontname='Arial', fontsize=12)
    axs[i].tick_params(axis='both', which='major', labelsize=12)  # Set font size for major tick labels
    axs[i].tick_params(axis='both', which='minor', labelsize=12)  # Set font size for minor tick labels


# Adjust layout
plt.tight_layout()

plt.savefig(csv_file_path[:-4]+"_Histograms_grid.pdf")
# Show the plot
plt.show()

fig, axs = plt.subplots(7, 2, figsize=(12, 18))

# Flatten the axs array to iterate over each subplot
axs = axs.flatten()

# Create a dictionary to store histogram data
hist_data = {}

# Iterate over DataFrame columns
for i, col in enumerate(df.columns):
    # Calculate histogram and bin edges
    hist, bins = np.histogram(df[col].dropna(), bins=50, range=[1, 100000])
    
    # Normalize histogram counts
    hist_normalized = hist / hist.sum()
    
    # Store histogram data in the dictionary
    hist_data[col] = hist_normalized

    # Plot normalized histogram
    plt.bar(bins[:-1], hist_normalized, width=(bins[1] - bins[0]), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel('Intensity of spirosome', fontname='Arial', fontsize=12)
    plt.ylabel('Normalised Frequency', fontname='Arial', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)  # Set font size for major tick labels
    plt.tick_params(axis='both', which='minor', labelsize=12)  # Set font size for minor tick labels
    plt.show()

# Convert histogram data dictionary to DataFrame
hist_df = pd.DataFrame.from_dict(hist_data, orient='index')

# Transpose the DataFrame
hist_df = hist_df.transpose()

# Insert the bins as the first column
hist_df.insert(0, 'Bins', bins[:-1])

hist_df.to_csv(csv_file_path[:-4]+'_histogram_data.csv')
# Adjust layout
plt.tight_layout()

plt.savefig(csv_file_path[:-4]+"_Histograms_norm_grid.pdf")
# Show the plot
plt.show()

