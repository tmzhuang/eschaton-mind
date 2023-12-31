import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

seed = 42
df = pd.read_csv('clean.csv')
print(df.head())

train, test = train_test_split(df)
y = df.tto
x = df.drop(['tto', 'fc', 'vq', 'ub', 'st', 'gy', 'tl', 'unk', 'mys', 'tto <= 5m'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)
rfr = RandomForestRegressor(random_state=seed)
rfr.fit(x_train, y_train)
print('Model training complete.')
y_pred = rfr.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm)
print(classification_report(y_test, y_pred))

joblib.dump(rfr, 'model.sav')
