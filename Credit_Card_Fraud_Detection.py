import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('fraudTrain.csv')
print(data.head(3))
#data preprocessing
print(data.isnull().sum())
data.info()
data
data = data.drop('Unnamed: 0',axis = 1)
print(data)
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
encoder = LabelEncoder()

# Select the categorical columns for encoding
categorical_columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'first', 'last','street','city','lat','long','job','dob','unix_time','merch_lat','merch_long','gender','state']

# Apply LabelEncoder to each categorical column
encoded_data = data[categorical_columns].apply(lambda col: encoder.fit_transform(col))

# Concatenate the encoded categorical columns with the remaining columns
encoded_data = pd.concat([encoded_data, data.drop(categorical_columns, axis=1)], axis=1)

# Now, 'encoded_data' contains the encoded categorical columns along with the remaining columns
print(encoded_data)

encoded_data = encoded_data.drop('trans_num',axis = 1)
print(encoded_data)
x = encoded_data.iloc[:,0:-1]
print(x)
y = encoded_data.iloc[:,-1]
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


feature_names = x.columns

from sklearn import tree, metrics
dtree=tree.DecisionTreeClassifier(criterion='gini')
dtree.fit(x_train,y_train)

y_pred = dtree.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
classification_rep = classification_report(y_test, y_pred)

print("Decision Tree Model:")
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_rep)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

print(accuracy_rf)
