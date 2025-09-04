# Create subplots for better organization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Visualizations', fontsize=16, fontweight='bold')

# 1. Line chart showing trends (using index as pseudo-time)
axes[0, 0].plot(iris_df.index[:50], iris_df['sepal length (cm)'][:50], 
                label='Sepal Length', marker='o', linewidth=2)
axes[0, 0].plot(iris_df.index[:50], iris_df['petal length (cm)'][:50], 
                label='Petal Length', marker='s', linewidth=2)
axes[0, 0].set_title('Trend of Sepal and Petal Length (First 50 Samples)')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Bar chart - average measurements by species
species_means = iris_df.groupby('species').mean()
x = np.arange(len(species_means.index))
width = 0.2

bars1 = axes[0, 1].bar(x - width, species_means['sepal length (cm)'], width, 
                       label='Sepal Length', alpha=0.8)
bars2 = axes[0, 1].bar(x, species_means['petal length (cm)'], width, 
                       label='Petal Length', alpha=0.8)
bars3 = axes[0, 1].bar(x + width, species_means['sepal width (cm)'], width, 
                       label='Sepal Width', alpha=0.8)

axes[0, 1].set_title('Average Measurements by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Measurement (cm)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(species_means.index)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Histogram - distribution of sepal length
axes[1, 0].hist(iris_df['sepal length (cm)'], bins=15, alpha=0.7, 
                color='skyblue', edgecolor='black')
axes[1, 0].set_title('Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot - sepal length vs petal length
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species, color in colors.items():
    species_data = iris_df[iris_df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'], 
                      alpha=0.7, label=species, c=color)

axes[1, 1].set_title('Sepal Length vs Petal Length by Species')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = iris_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Box plot to show distribution by species
plt.figure(figsize=(12, 6))
iris_df_melted = pd.melt(iris_df, id_vars="species", 
                        value_vars=iris_df.columns[:-1])
sns.boxplot(x="variable", y="value", hue="species", data=iris_df_melted)
plt.title('Distribution of Measurements by Species')
plt.xlabel('Measurement Type')
plt.ylabel('Measurement (cm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
