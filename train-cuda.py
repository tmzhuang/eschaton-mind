import cudf as pd
from cuml.ensemble import RandomForestRegressor
from cuml.model_selection import train_test_split
from cuml.metrics import accuracy_score
import pickle
seed = 42
df = pd.read_csv('clean.csv')

y = df['tto <= 5m'].astype('int8')
x = df.drop(['tto', 'tto <= 5m'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed)
breakpoint()
rfr = RandomForestRegressor(random_state=seed)
rfr.fit(x_train, y_train)
print('Model training complete.')
pickle.dump(rfr, open('model-boolean.pkl', 'wb'))

# accuracy
y_pred = rfr.predict(x_test)
print('accuracy:', accuracy_score(y_test, y_pred))
