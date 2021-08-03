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
def make_lerp_data(df):
    '''
    Generate interpolated lat,lng values for different phone times in the same collection.
    '''
    org_columns = df.columns

    # Generate a combination of time x collection x phone and combine it with the original data (generate records to be interpolated)
    time_list = df[['collectionName', 'millisSinceGpsEpoch']].drop_duplicates()
    phone_list =df[['collectionName', 'phoneName']].drop_duplicates()
    tmp = time_list.merge(phone_list, on='collectionName', how='outer')

    lerp_df = tmp.merge(df, on=['collectionName', 'millisSinceGpsEpoch', 'phoneName'], how='left')
    lerp_df['phone'] = lerp_df['collectionName'] + '_' + lerp_df['phoneName']
    lerp_df = lerp_df.sort_values(['phone', 'millisSinceGpsEpoch'])

    # linear interpolation
    lerp_df['latDeg_prev'] = lerp_df['latDeg'].shift(1)
    lerp_df['latDeg_next'] = lerp_df['latDeg'].shift(-1)
    lerp_df['lngDeg_prev'] = lerp_df['lngDeg'].shift(1)
    lerp_df['lngDeg_next'] = lerp_df['lngDeg'].shift(-1)
    lerp_df['phone_prev'] = lerp_df['phone'].shift(1)
    lerp_df['phone_next'] = lerp_df['phone'].shift(-1)
    lerp_df['time_prev'] = lerp_df['millisSinceGpsEpoch'].shift(1)
    lerp_df['time_next'] = lerp_df['millisSinceGpsEpoch'].shift(-1)
    # Leave only records to be interpolated
    lerp_df = lerp_df[(lerp_df['latDeg'].isnull())&(lerp_df['phone']==lerp_df['phone_prev'])&(lerp_df['phone']==lerp_df['phone_next'])].copy()
    # calc lerp
    lerp_df['latDeg'] = lerp_df['latDeg_prev'] + ((lerp_df['latDeg_next'] - lerp_df['latDeg_prev']) * ((lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev']))) 
    lerp_df['lngDeg'] = lerp_df['lngDeg_prev'] + ((lerp_df['lngDeg_next'] - lerp_df['lngDeg_prev']) * ((lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev']))) 

    # Leave only the data that has a complete set of previous and next data.
    lerp_df = lerp_df[~lerp_df['latDeg'].isnull()]

    return lerp_df[org_columns]
# %%
def calc_mean_pred(df, lerp_df):
    '''
    Make a prediction based on the average of the predictions of phones in the same collection.
    '''
    add_lerp = pd.concat([df, lerp_df])
    mean_pred_result = add_lerp.groupby(['collectionName', 'millisSinceGpsEpoch'])[['latDeg', 'lngDeg']].mean().reset_index()
    mean_pred_df = df[['collectionName', 'phoneName', 'millisSinceGpsEpoch']].copy()
    mean_pred_df = mean_pred_df.merge(mean_pred_result[['collectionName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']], on=['collectionName', 'millisSinceGpsEpoch'], how='left')
    return mean_pred_df

# %%
# df = pd.read_csv('../../data/interim/kalman_s2g_moving_or_not_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_moving_or_not_PAOnothing.csv') # 本来やる必要なかった
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_moving_or_not_PAOnothing.csv')
# df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman.csv')
df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_lgbm_kalman.csv')
df = pd.concat([df, df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
df
# %%
# mean_predict
df_lerp = make_lerp_data(df)
df_mean_pred = calc_mean_pred(df, df_lerp)
df_mean_pred

# %%
# visualize
def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,
                            # Here, plotly gets, (x,y) coordinates|
                            lat="latDeg",
                            lon="lngDeg",
                            #Here, plotly detects color of series
                            color="phoneName",
                            labels="phoneName",
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
                # '2021-04-21-US-MTV-1', 
                # '2021-04-28-US-MTV-2',  # Uターン？
                '2021-04-29-US-MTV-2'
                # '2021-03-16-US-RWC-2'
            ],
        # 'SVL': ['2021-04-26-US-SVL-2'],
    }

# %%
# SJC
visualize_collection(df, cns_dict['SJC'])
visualize_collection(df_mean_pred, cns_dict['SJC'])

# %%
# MTV
visualize_collection(df, cns_dict['MTV'])
visualize_collection(df_mean_pred, cns_dict['MTV'])

# %%
# Trueなら下のセル実行OK
sample_df = pd.read_csv('../../data/submission/sample_submission.csv')
sample_df = pd.concat([sample_df, sample_df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
(sample_df[['collectionName', 'phoneName']].drop_duplicates() == df_mean_pred[['collectionName', 'phoneName']].drop_duplicates()).all().all()
# %%
# sub
sub = pd.read_csv('../../data/submission/sample_submission.csv')
sub = sub.assign( latDeg=df_mean_pred['latDeg'], lngDeg=df_mean_pred['lngDeg'])
sub
# sub.to_csv('../../data/submission/kalman_s2g_monPAOnothing_mp_pm.csv', index=False)
# sub.to_csv('../../data/submission/imu_many_lat_lng_deg_monPAOnothing_mp_pm.csv', index=False)
sub.to_csv('../../data/interim/imu_many_lat_lng_deg_lgbm_kalman_mp.csv', index=False)
