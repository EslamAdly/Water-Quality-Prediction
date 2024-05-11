import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Read the dataset
df = pd.read_excel("processed_data.xlsx")

# Separate features (x) and target variable (y)
x = df.drop("Potability", axis=1)
y = df["Potability"]

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Random oversampling to deal with class imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(x, y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
