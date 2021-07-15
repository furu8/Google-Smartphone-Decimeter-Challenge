# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
%%time
phone_name = input('スマホの名前指定: ')
dir_name = input('ディレクトリの名前指定（train/test）: ')

basepath = '../../data/interim'
df = pd.read_csv(f'{basepath}/{dir_name}/groupbyed_{phone_name}.csv')
df
# %%
cols = df.columns
for col in cols:
    print(col)

# %%
df_list = [df[[col]] for col in cols if not df.loc[df[col]!=0, col].empty]

# %%
concated_df = pd.DataFrame()
for onedf in df_list:
    concated_df = pd.concat([concated_df, onedf], axis=1)

# %%
concated_df
# %%
%%time
# 可視化しきれん
for i, col in enumerate(concated_df.columns[1:]):
    print(i, col)
    plt.figure(figsize=(20,4))
    plt.plot(concated_df[col])
    plt.show()

# %%
columns = [
    'xSatPosM', 'ySatPosM', 'zSatPosM',
    'xSatVelMps', 'ySatVelMps', 'zSatVelMps', 'rawPrM',
    'rollDeg', 'pitchDeg', 'yawDeg'
]

for i, col in enumerate(columns):
    print(i, col)
    plt.figure(figsize=(20,4))
    plt.plot(concated_df[col])
    plt.show()
