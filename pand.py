import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the Iris dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Explore dataset structure
print("\nDataset structure:")
print(iris_df.info())

print("\nDataset shape:", iris_df.shape)

# Check for missing values
print("\nMissing values:")
print(iris_df.isnull().sum())

# Clean the dataset (though Iris dataset typically has no missing values)
# For demonstration, we'll show how to handle missing values
if iris_df.isnull().sum().sum() > 0:
    # Fill numerical columns with mean and categorical with mode
    for column in iris_df.columns:
        if iris_df[column].dtype in ['int64', 'float64']:
            iris_df[column].fillna(iris_df[column].mean(), inplace=True)
        else:
            iris_df[column].fillna(iris_df[column].mode()[0], inplace=True)
    print("Missing values handled!")
else:
    print("No missing values found - dataset is clean!")
