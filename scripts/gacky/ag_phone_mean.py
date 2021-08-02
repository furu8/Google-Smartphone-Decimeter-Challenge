# %%
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
# %%
def apply_gauss_smoothing(df, params):
    SZ_1 = params['sz_1']
    SZ_2 = params['sz_2']
    SZ_CRIT = params['sz_crit']

    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()

        lat_g1 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_1))
        lon_g1 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_1))
        lat_g2 = gaussian_filter1d(data[:, 0], np.sqrt(SZ_2))
        lon_g2 = gaussian_filter1d(data[:, 1], np.sqrt(SZ_2))

        lat_dif = data[1:,0] - data[:-1,0]
        lon_dif = data[1:,1] - data[:-1,1]

        lat_crit = np.append(np.abs(gaussian_filter1d(lat_dif, np.sqrt(SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lat_dif), np.sqrt(SZ_CRIT)))),[0])
        lon_crit = np.append(np.abs(gaussian_filter1d(lon_dif, np.sqrt(SZ_CRIT)) / (1e-9 + gaussian_filter1d(np.abs(lon_dif), np.sqrt(SZ_CRIT)))),[0])           

        df.loc[cond, 'latDeg'] = lat_g1 * lat_crit + lat_g2 * (1.0 - lat_crit)
        df.loc[cond, 'lngDeg'] = lon_g1 * lon_crit + lon_g2 * (1.0 - lon_crit)

    return df
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
train_base = pd.read_csv('../../data/raw/baseline_locations_train.csv')
sub = pd.read_csv('../../data/submission/sample_submission.csv')
# %%
train_base = pd.read_csv('../../data/interim/kalman.csv')
# %%
smoothed_baseline = apply_gauss_smoothing(train_base, {'sz_1' : 0.85, 'sz_2' : 5.65, 'sz_crit' : 1.5})
# %%
smoothed_baseline = mean_with_other_phones(smoothed_baseline)
# %%
smoothed_baseline.to_csv('../../data/interim/kalman_s2gt_SJC_mean_predict_phone_mean.csv', index=False)
# %%
smoothed_baseline = pd.read_csv('../../data/interim/kalman_s2gt_SJC_mean_predict.csv')
# %%
smoothed_baseline.isnull().sum()
# %%
smoothed_baseline
# %%
sub = sub.assign( latDeg=smoothed_baseline.latDeg, lngDeg=smoothed_baseline.lngDeg )
sub.to_csv('./submission19.csv', index=False)
# %%
