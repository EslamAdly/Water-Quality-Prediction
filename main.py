import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



df = pd.read_excel("processed_data.xlsx")
#print(df_rand.head())

x=df.drop("Potability", axis=1)
y=df["Potability"]

scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


def decisiontree():
    #create model object
    dt_model = DecisionTreeClassifier(max_depth=4)

    #train it 
    dt_model.fit(x_train, y_train)

    dt_prediction= dt_model.predict(x_test)

    dt_accuracy_score= accuracy_score(y_test, dt_prediction)

    print(dt_accuracy_score)

decisiontree()

