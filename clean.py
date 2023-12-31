''' Prepare data for training.

Add headers to data.
tto - time to open
fc..mys - key count
sod - second of day
dow - day of week
players - player count on server

We add some additional features for training.

fc_r, vq_r, etc. - ratio of keys to max
fr_d, vq_d, etc. - delta in key count over last n seconds

Output cleaned data as clean.csv.
'''
import pandas as pd

df = pd.read_csv('train.player_counts.2023-12-11.txt', sep=' ', header=None)
df.columns = ['tto', 'fc', 'vq', 'ub', 'st', 'gy', 'tl', 'unk', 'mys', 'sod', 'dow', 'players']
df['tto <= 5m'] = df.tto <= 600
df['fc_r'] = df.fc / 1400
df['vq_r'] = df.vq / 1400
df['ub_r'] = df.ub / 1200
df['st_r'] = df.st / 1200
df['gy_r'] = df.gy / 1000
df['st_r'] = df.st / 1000
df['unk_r'] = df.unk / 800
df['mys_r'] = df.mys / 800

# use n seconds to perform delta
n = 60
df1 = df.loc[n-1:, 'fc':'mys']
df2 = df.loc[:len(df) - n, 'fc':'mys']

df_d = df1.reset_index(drop=True) - df2.reset_index(drop=True)
df_d.columns = df_d.columns + '_d'
clean = pd.concat((df[n-1:].reset_index(drop=True), df_d.reset_index(drop=True)), axis=1)
clean
clean.to_csv('./clean.csv', index=False)
