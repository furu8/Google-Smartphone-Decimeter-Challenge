# %%
# import library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib_venn import venn2, venn2_circles
import seaborn as sns
from tqdm.notebook import tqdm
import pathlib
import plotly
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
def visualize_trafic(df, center, zoom=9):
    fig = px.scatter_mapbox(df,
                            # Here, plotly gets, (x,y) coordinates
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
def visualize_collection(df, collection):
    target_df = df[df['collectionName']==collection].copy()
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    visualize_trafic(target_df, center)
# %%
base_train = pd.read_csv('../../data/raw/baseline_locations_train.csv')
base_test = pd.read_csv('../../data/raw/baseline_locations_test.csv')
sample_sub = pd.read_csv('../../data/submission/sample_submission.csv')
# %%
sub_df = pd.read_csv('../../data/submission/sample_submission.csv')
# %%
# ground_truth
p = pathlib.Path('../../data/raw/')
gt_files = list(p.glob('train/*/*/ground_truth.csv'))
print('ground_truth.csv count : ', len(gt_files))

gts = []
for gt_file in tqdm(gt_files):
    gts.append(pd.read_csv(gt_file))
ground_truth = pd.concat(gts)

display(ground_truth.head())
# %%
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
# %%
# reject outlier
train_ro = add_distance_diff(base_train)
th = 50
train_ro.loc[((train_ro['dist_prev'] > th) & (train_ro['dist_next'] > th)), ['latDeg', 'lngDeg']] = np.nan
# %%
import simdkalman
# %%
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

def apply_kf_smoothing(df, kf_=kf):
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
cols = ['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']
train_ro_kf = apply_kf_smoothing(train_ro[cols])
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
test_kf = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_rfm_kalman_s2gt_SJC_thres4.csv')
test_kf
# %%
train_lerp = make_lerp_data(train_ro_kf)
train_mean_pred = calc_mean_pred(train_ro_kf, train_lerp)
# %%
tmp1 = train_ro_kf.copy()
tmp2 = train_mean_pred.copy()
tmp2['phoneName'] = tmp2['phoneName'] + '_MEAN'
tmp3 = ground_truth.copy()
tmp3['phoneName'] = tmp3['phoneName'] + '_GT'
tmp = pd.concat([tmp1, tmp2, tmp3])
visualize_collection(tmp, '2020-05-14-US-MTV-1')
# %%
def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)
# %%
def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2 
    score = res['p50_p90_mean'].mean()
    return score
# %%
print('kf + reject_outlier : ', get_train_score(train_ro_kf, ground_truth))
print('+ phones_mean_pred : ', get_train_score(train_mean_pred, ground_truth))
# %%
base_test = pd.read_csv('../../data/interim/kalman_s2g_mean_predict_phone_mean.csv')
# %%
base_test = add_distance_diff(base_test)
th = 50
base_test.loc[((base_test['dist_prev'] > th) & (base_test['dist_next'] > th)), ['latDeg', 'lngDeg']] = np.nan
#base_test.to_csv('../../data/interim/outlier_train.csv', index=False)
test_kf = apply_kf_smoothing(base_test)
# %%
test_kf.to_csv('../../data/interim/kalman_s2g_mean_predict_phone_mean_kalman.csv', index=False)
# %%
test_lerp = make_lerp_data(test_kf)
test_mean_pred = calc_mean_pred(test_kf, test_lerp)
# %%
sample_sub['latDeg'] = test_mean_pred['latDeg']
sample_sub['lngDeg'] = test_mean_pred['lngDeg']
#sample_sub.to_csv('submission4.csv', index=False)
# %%
test_mean_pred["heightAboveWgs84EllipsoidM"] = test_kf["heightAboveWgs84EllipsoidM"]
test_mean_pred["phone"] = test_kf["phone"]
# %%
base_test["heightAboveWgs84EllipsoidM"] = test_mean_pred["heightAboveWgs84EllipsoidM"]
base_test["phone"] = test_mean_pred["phone"]
base_test#test_mean_pred
# %%
test_mean_pred.to_csv('../../data/interim/imu_many_lat_lng_deg_rfm_kalman_s2gt_SJC_thres4_mean_predict.csv', index=False)
# %%
import plotly.express as px
# %%
fig = px.scatter_mapbox(test_mean_pred, #line_points, #kf_kf_smoothed_baseline[kf_kf_smoothed_baseline["phone"] == "2021-04-22-US-SJC-2_SamsungS20Ultra"],

                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg",
                    lon="lngDeg",

                    zoom=15,
                    center={"lat":37.33351, "lon":-121.8906},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()
# %%
sample_sub["lngDeg"].plot()
# %%
new_column = lambda x:x.split('_')
# %%
inter = pd.DataFrame(test_kf["phone"].apply(new_column))
# %%
test_mean_pred
# %%
sub_df
# %%
add_column = lambda x:x[1]
# %%
interrim["phoneName"] = pd.DataFrame(list(map(lambda x: add_column(x[1]), inter.itertuples())))
# %%
interrim = interrim.rename({0:"collectionName"}, axis=1)
# %%
test_kf = pd.concat([test_kf, interrim], axis=1)
# %%
best_df = pd.read_csv('./submission_adaptgauss_mean.csv')
# %%
sub_df = sub_df.assign(
    latDeg = sample_sub.latDeg,
    lngDeg = sample_sub.lngDeg
)
sub_df.to_csv('./submission16.csv', index=False)
# %%
base_train.describe()
# %%
