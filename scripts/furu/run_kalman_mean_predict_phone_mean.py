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

# 外れ値
def add_distance_diff(df):
    df['latDeg_prev'] = df['latDeg'].shift(1)
    df['latDeg_next'] = df['latDeg'].shift(-1)
    df['lngDeg_prev'] = df['lngDeg'].shift(1)
    df['lngDeg_next'] = df['lngDeg'].shift(-1)
    df['phone_prev'] = df['phone'].shift(1)
    df['phone_next'] = df['phone'].shift(-1)

    df['dist_prev'] = calc_haversine(df['latDeg'], df['lngDeg'], df['latDeg_prev'], df['lngDeg_prev'])
    df['dist_next'] = calc_haversine(df['latDeg'], df['lngDeg'], df['latDeg_next'], df['lngDeg_next'])

    df.loc[df['phone']!=df['phone_prev'], ['latDeg_prev', 'lngDeg_prev', 'dist_prev']] = np.nan
    df.loc[df['phone']!=df['phone_next'], ['latDeg_next', 'lngDeg_next', 'dist_next']] = np.nan

    return df

def apply_kf_smoothing(df, kf_):
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]
    return df
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
df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg.csv')
df

# %%
df = pd.concat([df, df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
df
# %%
# 外れ値
df_outlier = add_distance_diff(df)
th = 50
df_outlier.loc[((df_outlier['dist_prev'] > th) & (df_outlier['dist_next'] > th)), ['latDeg', 'lngDeg']] = np.nan
df_outlier

# %%
# カルマンフィルタ
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
df_kf = apply_kf_smoothing(df_outlier, kf)
df_kf
# %%
# mean_predict
df_lerp = make_lerp_data(df_kf)
df_mean_pred = calc_mean_pred(df_kf, df_lerp)
df_mean_pred

# %%
# phone_mean
df_phone_mean = mean_with_other_phones(df_mean_pred)
df_phone_mean

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
                '2021-04-21-US-MTV-1', 
                '2021-04-28-US-MTV-2', 
                '2021-04-29-US-MTV-2'
            ],
        # 'SVL': ['2021-04-26-US-SVL-2'],
    }

# %%
# SJC
visualize_collection(df_kf, cns_dict['SJC'])
visualize_collection(df_mean_pred, cns_dict['SJC'])
visualize_collection(df_phone_mean, cns_dict['SJC'])

# %%
# MTV
visualize_collection(df_kf, cns_dict['MTV'])
visualize_collection(df_mean_pred, cns_dict['MTV'])
visualize_collection(df_phone_mean, cns_dict['MTV'])
# %%
# sub
sub = pd.read_csv('../../data/submission/sample_submission.csv')
sub = sub.assign( latDeg=df_phone_mean['latDeg'], lngDeg=df_phone_mean['lngDeg'])
sub
sub.to_csv('../../data/submission/imu_many_lat_lng_deg_kalman_mean_predict_phone_mean.csv', index=False)