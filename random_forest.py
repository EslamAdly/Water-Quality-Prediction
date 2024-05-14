import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import GridSearchCV

# Read the dataset
df = pd.read_excel("processed_data.xlsx")

# Separate features (x) and target variable (y)
x = df.drop("Potability", axis=1)
y = df["Potability"]

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Random oversampling to deal with class imbalance
ros = SMOTEN(random_state=42)
X_resampled, y_resampled = ros.fit_resample(x, y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()


# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2']  
}

# Create the GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the grid search
grid_search.fit(x_train, y_train)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Print the best score found
print("Best Score:", grid_search.best_score_)

# Make predictions on the test set using the best estimator from grid search
y_pred = grid_search.best_estimator_.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)



# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
