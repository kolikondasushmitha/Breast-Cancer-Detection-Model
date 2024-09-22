# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
data = pd.read_csv('C:/Users/Venka/Downloads/Breast_Cancer_Dataset/breast-cancer.csv')

# Convert diagnosis column to numeric for classification (M = 1, B = 0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

### 1. Distribution of Diagnoses (Malignant vs. Benign)
diagnosis_counts = data['diagnosis'].value_counts()

# Plotting bar chart for diagnosis distribution
plt.figure(figsize=(6,4))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, palette=['skyblue', 'orange'])
plt.title("Distribution of Diagnoses (Malignant vs Benign)")
plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)")
plt.ylabel("Number of Cases")
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.show()

### 2. Correlation Heatmap
# Create a heatmap for correlations between features
plt.figure(figsize=(12,10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Tumor Features")
plt.show()

### 3. Boxplots for selected features
# Comparing 'radius_mean', 'area_mean', 'texture_mean', and 'smoothness_mean' for benign and malignant tumors
plt.figure(figsize=(15,10))
features_to_compare = ['radius_mean', 'area_mean', 'texture_mean', 'smoothness_mean']
for i, feature in enumerate(features_to_compare, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='diagnosis', y=feature, data=data, palette='Set2')
    plt.title(f'Boxplot of {feature} by Diagnosis')

plt.tight_layout()
plt.show()

### 4. Pair Plot for selected features
# Scatter plot matrix for 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean' with respect to diagnosis
selected_features = ['radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
sns.pairplot(data[selected_features], hue='diagnosis', palette='Set1')
plt.show()

### 5. Feature Importance using Random Forest
# Split dataset into features (X) and target (y)
X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis']

# Initialize and train RandomForest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]  # Get the top 10 important features

# Plot Feature Importances
plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Top 10 Important Features (Random Forest)')
plt.show()
