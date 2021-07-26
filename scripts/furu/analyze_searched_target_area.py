# %%
import numpy as np
from cv2 import Rodrigues
from numpy.core.numeric import ones
import pandas as pd
from pathlib import Path
from pandas.io.parsers import count_empty_vals
import pyproj
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from tqdm import tqdm
import glob as gb
import simdkalman
import re
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
def scoreing_dist(tr_df, cn_list, key_list):
    score_df = pd.DataFrame(cn_list, columns=['collectionName'])
    features = []

    for key in key_list:
        print(key)
        score_list = []
        for cn in cn_list:
            onecn_df = tr_df[tr_df['collectionName']==cn]
            lat = onecn_df['latDeg'].values
            lng = onecn_df['lngDeg'].values
            lat_pred = onecn_df[f'latDeg_{key}'].values
            lng_pred = onecn_df[f'lngDeg_{key}'].values
            
            onecn_df['dist'] = calc_haversine(lat, lng, lat_pred, lng_pred)

            dist_50 = np.percentile(onecn_df['dist'], 50)
            dist_95 = np.percentile(onecn_df['dist'], 95)
            dist_50_95 = (dist_50 + dist_95) / 2

            print(f'\n{cn}')
            print('dist_50:', dist_50)
            print('dist_95:', dist_95)
            print('avg_dist_50_95:', dist_50_95) 
            print('avg_dist:', onecn_df['dist'].mean())

            score_list.append(dist_50_95)

        features.append(f'latDeg_{key}')
        features.append(f'lngDeg_{key}')
        score_df[f'avg_dist50_95_{key}'] = score_list
    
    return score_df

    
# %%
tr_df = pd.read_csv('../../data/processed/train/train_merged_base.csv')[['millisSinceGpsEpoch', 'collectionName', 'phoneName', 'latDeg', 'lngDeg', 'latDeg_pred', 'lngDeg_pred']]
tr_df = tr_df.rename(columns={'latDeg_pred':'latDeg_base', 'lngDeg_pred':'lngDeg_base'})
tr_df

# %%
cn2pn_df = tr_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df = cn2pn_df.reset_index(drop=True)
cn2pn_df

# %%
cn_list = tr_df['collectionName'].unique()
cn_list
# %%
train_pahts = gb.glob('../../data/interim/*_train.csv')

latlng_dict = {}
for train_path in train_pahts:
    key = re.split('/|_train', train_path)[4]
    print(key)
    latlng_dict[key] = pd.read_csv(train_path)

# %%
# rename
for key in latlng_dict.keys():
    print(key)
    latlng_dict[key] = latlng_dict[key].rename(columns={'latDeg':f'latDeg_{key}', 'lngDeg':f'lngDeg_{key}'})
    display(latlng_dict[key])

# %%
# merge
for key, value in latlng_dict.items():
    print(key)
    display(value)
    tr_df = pd.merge_asof(
                    tr_df,
                    value[[f'latDeg_{key}', f'lngDeg_{key}', 'millisSinceGpsEpoch', 'collectionName', 'phoneName']].sort_values('millisSinceGpsEpoch'),
                    on='millisSinceGpsEpoch',
                    by=['collectionName', 'phoneName'],
                    direction='nearest'
    )
tr_df

# %%
score_df = scoreing_dist(tr_df, cn_list, latlng_dict.keys())
score_df
# %%
# Notion用
score_df['amari'] = 0
score_df
# %%
