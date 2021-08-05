# %%
import plotly.express as px
import pandas as pd
import numpy as np
import glob as gb
import re
from IPython.core.display import display

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
test_df = pd.read_csv(f'../../data/interim/train/train_org_moving_or_not_4interpolate.csv')
display(test_df)

# %%
test_df = pd.merge_asof(
            test_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg_bs', 'lngDeg_bs', 'tag']].sort_values('millisSinceGpsEpoch'),
            tr_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 
                    'latDeg', 'lngDeg', 
                    'latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean',
                    'latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict',
                    'latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict',	
                    'latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict',
                    'latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean',
                    'latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean',	
                    'latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict',	
                    'latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean']].sort_values('millisSinceGpsEpoch'),
            on='millisSinceGpsEpoch',
            by=['collectionName', 'phoneName'],
            direction='nearest'
)
test_df

# %%
def scatter_latlng(df):
    fig = px.scatter_mapbox(df,
                        # Here, plotly gets, (x,y) coordinates
                        lat="latDeg_kalman_mean_predict_phone_mean",
                        lon="lngDeg_kalman_mean_predict_phone_mean",
                        text='phoneName',

                        #Here, plotly detects color of series
                        color="tag",
                        labels="collectionName",

                        zoom=14.5,
                        center={"lat":37.334, "lon":-121.89},
                        height=600,
                        width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

# %%
cn2pn_df = test_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df

# %%
lat_lng_list = [['latDeg_kalman_mean_predict_phone_mean', 'lngDeg_kalman_mean_predict_phone_mean'],
['latDeg_aga_phone_mean_mean_predict', 'lngDeg_aga_phone_mean_mean_predict'],
['latDeg_kalman_mean_predict', 'lngDeg_kalman_mean_predict'],	
['latDeg_kalman_phone_mean_mean_predict', 'lngDeg_kalman_phone_mean_mean_predict'],
['latDeg_aga_mean_predict_phone_mean', 'lngDeg_aga_mean_predict_phone_mean'],
['latDeg_kalman_phone_mean', 'lngDeg_kalman_phone_mean'],	
['latDeg_aga_mean_predict', 'lngDeg_aga_mean_predict'],
['latDeg_aga_phone_mean', 'lngDeg_aga_phone_mean']]

for lat, lng in lat_lng_list:
    print(lat, lng)

# %%
%%time
new_test_df = test_df.copy()

for lat_col, lng_col in lat_lng_list:
    for cn, pn in cn2pn_df.values:
        idx = 0
        onedf = test_df[(test_df['collectionName']==cn) & (test_df['phoneName']==pn)].copy()
        idxes = onedf.index
        new_lat, new_lng = None, None
        first1_flag = False
        print(cn, pn)
        for lat, lng, tag in onedf[[lat_col, lng_col, 'tag']].values:
            if tag < 1:
                if first1_flag:
                    new_test_df.loc[idxes[:idx], lat_col] = lat
                    new_test_df.loc[idxes[:idx], lng_col] = lng
                    first1_flag = False
                new_lat = lat
                new_lng = lng
            elif tag == 1:
                if new_lat is None and new_lng is None:
                    first1_flag = True
                    pass
                else:
                    new_test_df.loc[idxes[idx], lat_col] = new_lat
                    new_test_df.loc[idxes[idx], lng_col] = new_lng
            idx += 1

# %%
display(test_df)
display(new_test_df)
display(new_test_df[new_test_df['collectionName']=='2021-03-16-US-RWC-2'])
display(new_test_df[new_test_df['collectionName']=='2021-04-29-US-SJC-3'])

# %%
scatter_latlng(new_test_df[new_test_df['collectionName']=='2021-04-29-US-SJC-3'])
# %%
# org
scatter_latlng(test_df)
# %%
# mon後
scatter_latlng(new_test_df)

# %%
new_test_df

# %%
cn_list = test_df['collectionName'].unique()
cn_list
# %%
score_df = scoreing_dist(new_test_df, cn_list, latlng_dict.keys())
score_df
# %%
score_df.sort_values('avg_dist50_95_kalman_mean_predict_phone_mean', ascending=False)
# %%
