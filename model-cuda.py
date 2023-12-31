import cudf as pd
import pickle
from cuml.metrics import accuracy_score, regression
from cuml.model_selection import train_test_split

rfr = pickle.load(open("model.pkl", "rb"))
#  rfr2 = pickle.load(open("model-2.pkl", "rb"))
#  rfrb = pickle.load(open("model-boolean.pkl", "rb"))

seed = 100
df = pd.read_csv('clean.csv')
y = df.tto
x1 = df.drop(['tto', 'fc', 'vq', 'ub', 'st', 'gy', 'tl', 'unk', 'mys', 'tto <= 5m'], axis=1)
#  x2 = df.drop(['tto', 'tto <= 5m'], axis=1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y, random_state=seed)
#  x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, df['tto <= 5m'], random_state=seed)
#  test = df[df.tto <= 1000].sample(n=20, random_state=seed)
pred = rfr.predict(x_test1)
print('rfr:', regression.r2_score(y_test1.astype('float32'), pred))
#  pred2 = rfr2.predict(x_test1)
#  print('rfr2:', regression.r2_score(y_test1.astype('float32'), pred2))
#  print('rfr2:', regression.r2_score(y_test2, pred))
#  predb = rfrb.predict(x_test2)
#  print('rfrb:', accuracy_score(y_test2, predb.astype('bool')))
#  results = pd.concat(dict(act=test.tto,
                         #  actb=test['tto <= 5m'],
                         #  pred=pred,
                         #  pred2=pred2,
                         #  predb=predb), axis=1)
#  print(results)
