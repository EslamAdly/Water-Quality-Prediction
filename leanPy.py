import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , accuracy_score
from imblearn.over_sampling import SMOTEN

df =pd.read_excel("processed_data.xlsx")

X = df.drop(columns = 'Potability')
Y = df['Potability']
ros = SMOTEN()
X_resampled, y_resampled = ros.fit_resample(X, Y)

X_Train,X_Test,Y_Train,Y_Test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=53)


# standardizing data set to improve accuracy
Scaler = StandardScaler()
X_train_scaled  = Scaler.fit_transform(X_Train)
X_test_scaled = Scaler.transform(X_Test)

model = LogisticRegression()
log = model.fit(X_train_scaled,Y_Train)


y_pred = model.predict(X_test_scaled)

print("accuracy: ",accuracy_score(Y_Test,y_pred))

print(confusion_matrix(Y_Test,y_pred))