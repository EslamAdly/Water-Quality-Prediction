# Import scikit-learn dataset library
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

# Load the dataset from the Excel file
data = pd.read_excel("processed_data.xlsx")

# Print the names of the features
print("Features: ", data.columns)

# Print the top 5 records of the data
# print(data.head())

# Import train_test_split function

# Define the features and target variables
X = data.drop(columns=["Potability"])  # Assuming 'potability' is the target column
y = data["Potability"]

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)

# Import svm model

# Create an SVM Classifier with a linear kernel
clf = svm.SVC(kernel="linear")

# Train the model using the training set
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation

# Calculate the accuracy of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the precision of the model
precision = metrics.precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate the recall of the model
recall = metrics.recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate the confusion matrix of the model
confusion_mat = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
