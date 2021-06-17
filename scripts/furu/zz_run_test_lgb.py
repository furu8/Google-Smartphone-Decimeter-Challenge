# %%
import optuna.integration.lightgbm as lgb

from models import ModelLGB 
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display
from models import Util
#最大表示行数を設定
pd.set_option('display.max_rows', 500)
import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# %%
df_train = pd.read_csv('../../data/interim/train/cv_pixel4.csv')

display(df_train.shape)
display(df_train.head())
display(df_train.describe())
# %%
df_train.info()

# %%
X = df_train.drop(['latDeg', 'lngDeg'], axis=1)
y = df_train[['millisSinceGpsEpoch', 'latDeg', 'lngDeg']]
display(X)
display(y)
# %%
X_mean = X.groupby('millisSinceGpsEpoch', as_index=False).mean()
X_mean

# %%
y_min = y.groupby('millisSinceGpsEpoch', as_index=False).min() # kuso-code
y_min

# %%
from math import radians, cos, sin, asin, sqrt
def lat_lon_dist(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    dist_list = []
    for i in range(df.shape[0]):
        lat1 = df["lat_truth"][i]
        lon1 = df["lng_truth"][i]
        lat2 = df["lat_pred"][i]
        lon2 = df["lng_pred"][i]
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers is 6371
        mdist = 6371* c*1000
        dist_list.append(mdist)
    
    return dist_list

def runner(X, y, run_name, params, n_fold=3):
    # データの並び順を元に分割する
    folds = TimeSeriesSplit(n_splits=n_fold)

    # fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    score_df_list = []
    # 学習用のデータとテスト用のデータに分割するためのインデックス情報を得る
    for i, (tr_idx, va_idx) in enumerate(folds.split(X, y)):
        score_df = pd.DataFrame()
        tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
        va_x, va_y = X.iloc[va_idx], y.iloc[va_idx]

        lat_model = ModelLGB(run_name, params)
        lat_model.train(tr_x, tr_y['latDeg'])
        lat_pred = lat_model.predict(va_x)

        lng_model = ModelLGB(run_name, params)
        lng_model.train(tr_x, tr_y['lngDeg'])
        lng_pred = lng_model.predict(va_x)

        score_df['lat_pred'] = lat_pred 
        score_df['lng_pred'] = lng_pred 
        score_df['lat_truth'] = va_y['latDeg'].values
        score_df['lng_truth'] = va_y['lngDeg'].values
        score_df['dist'] = lat_lon_dist(score_df)

        score_df_list.append(score_df)

    #     # 生のインデックス
    #     print(f'index of train: {train_index}')
    #     print(f'index of test: {test_index}')
    #     print('----------')
    #     # 元のデータを描く
    #     sns.lineplot(data=X, x='millisSinceGpsEpoch', y='xSatVelMps', ax=axes[i], label='original')
    #     # 学習用データを描く
    #     sns.lineplot(data=X.iloc[train_index], x='millisSinceGpsEpoch', y='xSatVelMps', ax=axes[i], label='train')
    #     # テスト用データを描く
    #     sns.lineplot(data=X.iloc[test_index], x='millisSinceGpsEpoch', y='xSatVelMps', ax=axes[i], label='test')

    # plt.show()
    return score_df_list

# %%
params = {
        "num_leaves" : 31,
        "learning_rate" : 0.1,
        'objective': 'regression',
        'metric' : 'rmse',
        'verbose' : -1
    }

score_df_list = runner(X_mean, y_min, 'lgb', params)
for score_df in score_df_list:
    eval_df = pd.DataFrame()
    dist50 = np.percentile(score_df['dist'], 50)
    dist95 = np.percentile(score_df['dist'], 95)
    print('-------')
    print(dist50)
    print(dist95)
    print((dist50 + dist95) / 2)
    
    # eval_df['dist50'] = np.percentile(score_df['dist'], 50)
    # eval_df['dist95'] = np.percentile(score_df['dist'], 95)
    # eval_df['avg_dist'] = np.mean(np.array(eval_df[['dist50', 'dist95']]), axis=1)
    # print("Val evaluation details:\n", eval_df)
# %%
