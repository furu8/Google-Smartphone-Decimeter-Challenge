# %%
import pandas as pd
import numpy as np
import simdkalman
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import plotly.express as px

# %%
def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

# %%
def mean_with_other_phones(df):
    collections_list = df[['collectionName']].drop_duplicates().to_numpy()

    for collection in collections_list:
        phone_list = df[df['collectionName'].to_list() == collection][['phoneName']].drop_duplicates().to_numpy()

        phone_data = {}
        corrections = {}
        for phone in phone_list:
            cond = np.logical_and(df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            phone_data[phone[0]] = df[cond][['millisSinceGpsEpoch', 'latDeg', 'lngDeg']].to_numpy()

        for current in phone_data:
            correction = np.ones(phone_data[current].shape, dtype=np.float)
            correction[:,1:] = phone_data[current][:,1:]
            
            # Telephones data don't complitely match by time, so - interpolate.
            for other in phone_data:
                if other == current:
                    continue

                loc = interp1d(phone_data[other][:,0], 
                               phone_data[other][:,1:], 
                               axis=0, 
                               kind='linear', 
                               copy=False, 
                               bounds_error=None, 
                               fill_value='extrapolate', 
                               assume_sorted=True)
                
                start_idx = 0
                stop_idx = 0
                for idx, val in enumerate(phone_data[current][:,0]):
                    if val < phone_data[other][0,0]:
                        start_idx = idx
                    if val < phone_data[other][-1,0]:
                        stop_idx = idx

                if stop_idx - start_idx > 0:
                    correction[start_idx:stop_idx,0] += 1
                    correction[start_idx:stop_idx,1:] += loc(phone_data[current][start_idx:stop_idx,0])                    

            correction[:,1] /= correction[:,0]
            correction[:,2] /= correction[:,0]
            
            corrections[current] = correction.copy()
        
        for phone in phone_list:
            cond = np.logical_and(df['collectionName'] == collection[0], df['phoneName'] == phone[0]).to_list()
            
            df.loc[cond, ['latDeg', 'lngDeg']] = corrections[phone[0]][:,1:]            
            
    return df

# %%
# df = pd.read_csv('../../data/interim/kalman_s2g_moving_or_not_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_moving_or_not_PAOnothing.csv') # 本来やる必要なかった
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_moving_or_not_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_mp_moving_or_not_kai_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_mp_moving_or_not_kai_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_s2gt_SJC_mean_predict_moving_or_not_kai_PAOnothing.csv')
df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_lgbm_kalman_mp_moving_or_not_kai_PAOnothing.csv')
df = pd.concat([df, df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
df

# %%
# phone_mean
new_df = df.copy()
df_phone_mean = mean_with_other_phones(new_df)
df_phone_mean

# %%
# visualize
def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,
                            # Here, plotly gets, (x,y) coordinates|
                            lat="latDeg",
                            lon="lngDeg",
                            #Here, plotly detects color of series
                            # color="phoneName",
                            # labels="phoneName",
                            zoom=zoom,
                            center=center,
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()
# %%
def visualize_collection(df, collections):
    target_df = df[df['collectionName'].isin(collections)].copy()
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    visualize_trafic(target_df, center)

# %%
cns_dict = {
        'SJC': [
                # '2021-04-02-US-SJC-1', # 場所が違う
                '2021-04-22-US-SJC-2', 
                '2021-04-29-US-SJC-3'
            ],
        'MTV': [
                # '2021-03-16-US-MTV-2',
                # '2021-04-08-US-MTV-1', 
                '2021-04-21-US-MTV-1', 
                '2021-04-28-US-MTV-2',  # Uターン？
                '2021-04-29-US-MTV-2'
                '2021-03-16-US-RWC-2'
            ],
        # 'SVL': ['2021-04-26-US-SVL-2'],
    }

# %%
# SJC
visualize_collection(df, cns_dict['SJC'])
visualize_collection(df_phone_mean, cns_dict['SJC'])

# %%
# MTV
visualize_collection(df, cns_dict['MTV'])
visualize_collection(df_phone_mean, cns_dict['MTV'])

# %%
# Trueなら下のセル実行OK
sample_df = pd.read_csv('../../data/submission/sample_submission.csv')
sample_df = pd.concat([sample_df, sample_df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
(sample_df[['collectionName', 'phoneName']].drop_duplicates() == df_phone_mean[['collectionName', 'phoneName']].drop_duplicates()).all().all()
# %%
# sub
sub = pd.read_csv('../../data/submission/sample_submission.csv')
sub = sub.assign( latDeg=df_phone_mean['latDeg'], lngDeg=df_phone_mean['lngDeg'])
sub
# sub.to_csv('../../data/submission/kalman_s2g_monPAOnothing_mp_pm.csv', index=False)
# sub.to_csv('../../data/submission/imu_many_lat_lng_deg_monPAOnothing_mp_pm.csv', index=False)
# sub.to_csv('../../data/submission/imu_many_lat_lng_deg_kalman_monPAOnothing_mp_pm.csv', index=False)
# sub.to_csv('../../data/submission/imu_many_lat_lng_deg_kalman_mp_monPAOnothing_pm.csv', index=False)
sub.to_csv('../../data/submission/imu_many_lat_lng_deg_lgbm_kalman_mp_monPAOnothing_pm.csv', index=False)
