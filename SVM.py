import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the dataset from the Excel file
data =pd.read_excel("processed_data.xlsx")


# Print the names of the features
print("Features: ", data.columns)

# Print the top 5 records of the data
print(data.head())

# Define the features and target variables
X = data.drop(columns=['Potability'])  # Assuming 'potability' is the target column
y = data['Potability']

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

# Apply feature scaling to the training and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM Classifier with an RBF kernel
clf = SVC(kernel='rbf')

# Perform hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model using the best hyperparameters and scaled data
best_clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
best_clf.fit(X_train_scaled, y_train)

# Predict the response for the test dataset
y_pred = best_clf.predict(X_test_scaled)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the confusion matrix of the model
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)