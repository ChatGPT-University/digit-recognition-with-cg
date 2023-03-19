# predictions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = train.drop(['label'], axis=1).values
y = train['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix
print(len(X_test[0]))
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# exporting
import joblib
filename = 'model.joblib'
joblib.dump(clf, filename)
