# %%
import numpy as np
from cv2 import Rodrigues
import pandas as pd
from pathlib import Path
import pyproj
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from tqdm import tqdm
import simdkalman
import warnings
warnings.filterwarnings("ignore", category=Warning)

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
# From：https://www.kaggle.com/emaerthin/demonstration-of-the-kalman-filter
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(a**0.5)
    dist = 6_367_000 * c
    return dist
# %%
tr_df = pd.read_csv('../../data/processed/train/train_merged_base.csv')[['millisSinceGpsEpoch', 'collectionName', 'phoneName', 'latDeg', 'lngDeg', 'latDeg_pred', 'lngDeg_pred']]
tr_df

# %%
cn2pn_df = tr_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df = cn2pn_df.reset_index(drop=True)
cn2pn_df

# %%
cn_list = tr_df['collectionName'].unique()
cn_list

# %%
# dist_list = []
# for cn in cn_list:
#     lat = tr_df.loc[tr_df['collectionName']==cn, 'latDeg'].values
#     lng = tr_df.loc[tr_df['collectionName']==cn, 'lngDeg'].values
#     lat_pred = tr_df.loc[tr_df['collectionName']==cn, 'latDeg_pred'].values
#     lng_pred = tr_df.loc[tr_df['collectionName']==cn, 'lngDeg_pred'].values

#     dist_list.append(calc_haversine(lat, lng, lat_pred, lng_pred))


# %%
lat = tr_df['latDeg'].values
lng = tr_df['lngDeg'].values
lat_pred = tr_df['latDeg_pred'].values
lng_pred = tr_df['lngDeg_pred'].values

tr_df['dist'] = calc_haversine(lat, lng, lat_pred, lng_pred)
tr_df

# %%
print('dist_50:',np.percentile(tr_df['dist'],50) )
print('dist_95:',np.percentile(tr_df['dist'],95) )
print('avg_dist_50_95:',(np.percentile(tr_df['dist'],50) + np.percentile(tr_df['dist'],95))/2)
print('avg_dist:', tr_df['dist'].mean())

# %%
tr_df.describe()

# %%
tr_df['dist'].hist()

# %%
score_list = []
std_list = []
for cn in cn_list:
    onecn_df = tr_df[tr_df['collectionName']==cn]
    lat = onecn_df['latDeg'].values
    lng = onecn_df['lngDeg'].values
    lat_pred = onecn_df['latDeg_pred'].values
    lng_pred = onecn_df['lngDeg_pred'].values
    onecn_df['dist'] = calc_haversine(lat, lng, lat_pred, lng_pred)

    print(f'\n{cn}')
    print('dist_50:',np.percentile(onecn_df['dist'],50) )
    print('dist_95:',np.percentile(onecn_df['dist'],95) )
    print('avg_dist_50_95:',(np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    print('avg_dist:', onecn_df['dist'].mean())

    score_list.append((np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    std_list.append(lat.std())

# %%
score_df = pd.DataFrame(score_list, columns=['avg_dist_50_95'])
score_df['collectionName'] = cn_list
score_df['std'] = std_list
score_df['a'] = 0
score_df
# %%
print(score_df)
# %%
score_df['avg_dist_50_95'].hist()
# %%
score_df[['avg_dist_50_95', 'std']].corr()

# %%
pn_list = tr_df['phoneName'].unique()
pn_list

# %%
score_list = []
std_list = []
for cn, pn in cn2pn_df.values:
    onecn_df = tr_df[(tr_df['collectionName']==cn)&(tr_df['phoneName']==pn)]
   
    lat = onecn_df['latDeg'].values
    lng = onecn_df['lngDeg'].values
    lat_pred = onecn_df['latDeg_pred'].values
    lng_pred = onecn_df['lngDeg_pred'].values
    onecn_df['dist'] = calc_haversine(lat, lng, lat_pred, lng_pred)

    print(f'\n{cn}, {pn}')
    print('dist_50:',np.percentile(onecn_df['dist'],50) )
    print('dist_95:',np.percentile(onecn_df['dist'],95) )
    print('avg_dist_50_95:',(np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    print('avg_dist:', onecn_df['dist'].mean())

    score_list.append((np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    std_list.append(lat.std())

# %%
cn2pn_df['avg_dist_50_90'] = score_list
cn2pn_df['std'] = std_list
cn2pn_df

# %%
cn2pn_df['avg_dist_50_90'].hist()

# %%
def kalman():
    T = 1.0
    state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], [0, 1, 0, T, 0, 0.5 * T ** 2], [0, 0, 1, 0, T, 0],
                                [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
    observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
    observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9

    kf = simdkalman.KalmanFilter(
            state_transition = state_transition,
            process_noise = process_noise,
            observation_model = observation_model,
            observation_noise = observation_noise)
    return kf

def apply_kf_smoothing(df, kf_):
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in tqdm(unique_paths):
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg_pred', 'lngDeg_pred']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg_pred'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg_pred'] = smoothed.states.mean[0, :, 1]
    return df

# %%
kf = kalman()
kalmaned_df = apply_kf_smoothing(tr_df, kf)
kalmaned_df
# %%
score_list = []
std_list = []
for cn, pn in cn2pn_df[['collectionName', 'phoneName']].values:
    onecn_df = kalmaned_df[(kalmaned_df['collectionName']==cn)&(kalmaned_df['phoneName']==pn)]
   
    lat = onecn_df['latDeg'].values
    lng = onecn_df['lngDeg'].values
    lat_pred = onecn_df['latDeg_pred'].values
    lng_pred = onecn_df['lngDeg_pred'].values
    onecn_df['dist'] = calc_haversine(lat, lng, lat_pred, lng_pred)

    print(f'\n{cn}, {pn}')
    print('dist_50:',np.percentile(onecn_df['dist'],50) )
    print('dist_95:',np.percentile(onecn_df['dist'],95) )
    print('avg_dist_50_95:',(np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    print('avg_dist:', onecn_df['dist'].mean())

    score_list.append((np.percentile(onecn_df['dist'],50) + np.percentile(onecn_df['dist'],95))/2)
    std_list.append(lat.std())

# %%
cn2pn_df['avg_dist_50_90_kalman'] = score_list
cn2pn_df['std_kalman'] = std_list
cn2pn_df

# %%
cn2pn_df.loc[cn2pn_df['avg_dist_50_90']>5.0, ['collectionName', 'phoneName', 'avg_dist_50_90', 'avg_dist_50_90_kalman', 'd']]