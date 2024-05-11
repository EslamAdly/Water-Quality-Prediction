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

x_train, x_test, y_train, y_test=train_test_split(X_resampled, y_resampled, test_size=0.2)



def decisiontree_best_params():
    # Best parameters found by GridSearchCV
    best_params = {'ccp_alpha': 0.0, 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2}

    # Create the DecisionTreeClassifier with the best parameters
    best_dt_model = DecisionTreeClassifier(**best_params, random_state=42)

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

decisiontree_best_params()
