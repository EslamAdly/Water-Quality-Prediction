import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler


df = pd.read_excel("processed_data.xlsx")

#print(df_rand.head())

x=df.drop("Potability", axis=1)
y=df["Potability"]

scaler=StandardScaler()
x=scaler.fit_transform(x)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(x, y)

x_train, x_test, y_train, y_test=train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)



from sklearn.model_selection import GridSearchCV

def descision_tree_model():
    # Define the parameter grid to search
    param_grid = {
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Instantiate the DecisionTreeClassifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)

    # Get the best estimator
    best_dt_model = grid_search.best_estimator_

    # Train the model
    best_dt_model.fit(x_train, y_train)

    # Predict using the best estimator
    dt_prediction = best_dt_model.predict(x_test)

    # Calculate accuracy
    dt_accuracy_score = accuracy_score(y_test, dt_prediction)
    print("Accuracy:", dt_accuracy_score)

    # Calculate the confusion matrix
    confusion_mat = confusion_matrix(y_test, dt_prediction)
    print("Confusion Matrix:")
    print(confusion_mat)

descision_tree_model()
