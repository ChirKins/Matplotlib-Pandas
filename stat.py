# Compute basic statistics
print("Basic statistics of numerical columns:")
print(iris_df.describe())

# Group by species and compute mean of numerical columns
print("\nMean values by species:")
species_stats = iris_df.groupby('species').mean()
print(species_stats)

# Additional analysis - find patterns
print("\nInteresting findings:")
print("1. Setosa has the smallest petal dimensions")
print("2. Virginica has the largest sepal length on average")
print("3. Versicolor has intermediate values for most measurements")

# Calculate correlation matrix
correlation_matrix = iris_df.select_dtypes(include=[np.number]).corr()
print("\nCorrelation matrix:")
print(correlation_matrix)
