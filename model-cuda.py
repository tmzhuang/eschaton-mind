import pandas as pd
import pickle
from cuml.metrics import accuracy_score

rfr = pickle.load(open("model.pkl", "rb"))
rfr2 = pickle.load(open("model-2.pkl", "rb"))
rfrb = pickle.load(open("model-boolean.pkl", "rb"))

seed = 100
df = pd.read_csv('clean.csv')
y = df.tto
x1 = df.drop(['tto', 'fc', 'vq', 'ub', 'st', 'gy', 'tl', 'unk', 'mys', 'tto <= 5m'], axis=1)
x2 = df.drop(['tto', 'tto <= 5m'], axis=1)

test = df[df.tto <= 1000].sample(n=20, random_state=seed)
pred = rfr.predict(test)
pred2 = rfr2.predict(test)
predb = rfrb.predict(test)
results = pd.concat(dict(act=test.tto,
                         actb=test['tto <= 5m'],
                         pred=pred,
                         pred2=pred2,
                         predb=predb), axis=1)
print(results)
