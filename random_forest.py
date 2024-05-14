import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Read the dataset
df = pd.read_excel("processed_data.xlsx")

# Separate features (x) and target variable (y)
x = df.drop("Potability", axis=1)
y = df["Potability"]

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# SMOTE to deal with class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(x, y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Train the initial model on the training data
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
