import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import GridSearchCV


df = pd.read_excel("processed_data.xlsx")
#print(df_rand.head())

x=df.drop("Potability", axis=1)
y=df["Potability"]

scaler=StandardScaler()
x=scaler.fit_transform(x)

#oversampling tecnique to overcome imbalance problem 
ros = SMOTEN(random_state=42)
x_resampled, y_resampled = ros.fit_resample(x, y)

x_train, x_test, y_train, y_test=train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=109)

#smote = SMOTEN(random_state=42)
#x_train, y_train = smote.fit_resample(x_train, y_train)



from sklearn.metrics import classification_report

def grid_search_decision_tree():
    # Define the parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],  
        'max_depth': [None, 10, 20, 30, 40, 50],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'max_features': [None, 'sqrt', 'log2']  
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Perform the grid search
    grid_search.fit(x_train, y_train)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)

    # Print the best score found
    print("Best Score:", grid_search.best_score_)

    # Predict using the best estimator
    dt_prediction = grid_search.predict(x_test)


    accuracy = accuracy_score(y_test, dt_prediction)
    print("Accuracy:", accuracy)


    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, dt_prediction))

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, dt_prediction))

grid_search_decision_tree()
