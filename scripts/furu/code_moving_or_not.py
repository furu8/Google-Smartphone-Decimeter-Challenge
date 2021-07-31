# %%
import pandas as pd
import numpy as np
import glob as gb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plotly.express as px
# %%
#  making ground truth file
def make_gt(collectionName, phoneName):
    # ground_truth
    gt_files = gb.glob('../../data/raw/train/*/*/ground_truth.csv')

    gts = []
    for gt_file in gt_files:
        gts.append(pd.read_csv(gt_file))
    ground_truth = pd.concat(gts)
    
    # baseline
    cols = ['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']
    baseline = pd.read_csv('../../data/raw/baseline_locations_train.csv', usecols=cols)
    ground_truth = ground_truth.merge(baseline, how='inner', on=cols[:3], suffixes=('_gt', '_bs'))
    ground_truth["millisSinceGpsEpoch"] = ground_truth["millisSinceGpsEpoch"]
    # ground_truth["millisSinceGpsEpoch"] = ground_truth["millisSinceGpsEpoch"]//1000
    if (collectionName is None) or (phoneName is None):
        return ground_truth
    else:
        return ground_truth[(ground_truth['collectionName'] == collectionName) & (ground_truth['phoneName'] == phoneName)]

# %%
def make_tag(df, tag_v):
    df.loc[df['speedMps'] < tag_v, 'tag'] = 1
    df.loc[df['speedMps'] >= tag_v, 'tag'] = 0
    return df

# %%
def add_IMU(df, dataset_name, cname, pname):
    # load GNSS log
    acc_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalAccel_add_columns.csv')
    gyr_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalGyro_add_columns.csv')
    mag_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_UncalMag_add_columns.csv')
    # ort_df = pd.read_csv(f'../../data/interim/{dataset_name}/merged_{pname}_OrientationDeg_add_columns.csv')

    acc_df = acc_df[acc_df['collectionName']==cname].reset_index(drop=True)
    gyr_df = gyr_df[gyr_df['collectionName']==cname].reset_index(drop=True)
    mag_df = mag_df[mag_df['collectionName']==cname].reset_index(drop=True)
    # ort_df = ort_df[ort_df['collectionName']==cname].reset_index(drop=True)

    acc_df['millisSinceGpsEpoch'] = acc_df['millisSinceGpsEpoch'].astype(np.int64)
    gyr_df['millisSinceGpsEpoch'] = gyr_df['millisSinceGpsEpoch'].astype(np.int64)
    mag_df['millisSinceGpsEpoch'] = mag_df['millisSinceGpsEpoch'].astype(np.int64)
    # ort_df['millisSinceGpsEpoch'] = ort_df['millisSinceGpsEpoch'].astype(np.int64)
    display(acc_df)
    acc_df["x_f_acce"] = acc_df["UncalAccelZMps2"]
    acc_df["y_f_acce"] = acc_df["UncalAccelXMps2"]
    acc_df["z_f_acce"] = acc_df["UncalAccelYMps2"]
    # magn 
    mag_df["x_f_magn"] = mag_df["UncalMagZMicroT"]
    mag_df["y_f_magn"] = mag_df["UncalMagYMicroT"]
    mag_df["z_f_magn"] = mag_df["UncalMagXMicroT"]
    # gyro
    gyr_df["x_f_gyro"] = gyr_df["UncalGyroXRadPerSec"]
    gyr_df["y_f_gyro"] = gyr_df["UncalGyroYRadPerSec"]
    gyr_df["z_f_gyro"] = gyr_df["UncalGyroZRadPerSec"] 

    df_merge = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps"]].sort_values('millisSinceGpsEpoch'), acc_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), mag_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyr_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')

    return df_merge

    # return acc_df, gyr_df, mag_df

def merge_df(df, acc_df, gyr_df, mag_df):
    df_merge = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps"]].sort_values('millisSinceGpsEpoch'), acc_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), mag_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyr_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')

    return df_merge

# %%
def make_train(train_cname, tag_v):
    # make ground_truth file
    gt = make_gt(None, None)
    train_df = pd.DataFrame()
    for cname in train_cname:
        phone_list = gt[gt['collectionName'] == cname]['phoneName'].drop_duplicates()
        for pname in phone_list:
            df = gt[(gt['collectionName'] == cname) & (gt['phoneName'] == pname)]
            # acc_df, gyr_df, mag_df = add_IMU('train', cname, pname)
            df = add_IMU(df, 'train', cname, pname)
            # df = merge_df(df, acc_df, gyr_df, mag_df)
            train_df = pd.concat([train_df, df])
    # make tag
    train_df = make_tag(train_df, tag_v)
    return train_df

# %%
def lgbm(train, test, col, lgb_params):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(train[col], train['tag'])
    preds = model.predict(test[col])
    print('confusion matrix :  \n', confusion_matrix(preds, test['tag']))
    print('accuracy score : ', accuracy_score(preds, test['tag']))
    return preds

# %%
def get_train_score(df):
    # calc_distance_error
    df['err'] =  calc_haversine(df.latDeg_bs, df.lngDeg_bs, 
    df.latDeg_gt, df.lngDeg_gt)
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2 
    score = res['p50_p90_mean'].mean()
    return score


def percentile50(x):
    return np.percentile(x, 50)


def percentile95(x):
    return np.percentile(x, 95)


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
train_cname = ['2021-04-29-US-SJC-2', '2021-03-10-US-SVL-1']
test_cname = ['2021-04-28-US-SJC-1']
tag_v = 0.5
col = ["x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn", "x_f_gyro", "y_f_gyro", "z_f_gyro"]

# parameter
lgb_params = {
    'num_leaves': 90,
    'n_estimators': 125,
}

# %%
# make train&test
train_df = make_train(train_cname, tag_v)
test_df = make_train(test_cname, tag_v)

# %%
display(train_df)
display(test_df)

# %%
display(train_df[['collectionName', 'phoneName']].drop_duplicates())
display(test_df[['collectionName', 'phoneName']].drop_duplicates())

# %%
sjc_pi4_trn = train_df[(train_df['collectionName']=='2021-04-29-US-SJC-2') & (train_df['phoneName']=='Pixel4')]
sjc_sam_trn = train_df[(train_df['collectionName']=='2021-04-29-US-SJC-2') & (train_df['phoneName']=='SamsungS20Ultra')]
svl_pi4xl_trn = train_df[(train_df['collectionName']=='2021-03-10-US-SVL-1') & (train_df['phoneName']=='Pixel4XL')]
svl_sam_trn = train_df[(train_df['collectionName']=='2021-03-10-US-SVL-1') & (train_df['phoneName']=='SamsungS20Ultra')]
sjc_pi4_tst = test_df[(test_df['collectionName']=='2021-04-28-US-SJC-1') & (test_df['phoneName']=='Pixel4')]
sjc_sam_tst = test_df[(test_df['collectionName']=='2021-04-28-US-SJC-1') & (test_df['phoneName']=='SamsungS20Ultra')]

# %%
new_train_df = pd.concat([sjc_sam_trn, sjc_pi4_trn])
new_train_df = pd.concat([new_train_df, svl_sam_trn])
new_train_df = pd.concat([new_train_df, svl_pi4xl_trn])
new_train_df

# %%
new_test_df = pd.concat([sjc_sam_tst, sjc_pi4_tst])
new_test_df

# %%
display(new_train_df[['collectionName', 'phoneName']].drop_duplicates())
display(new_test_df[['collectionName', 'phoneName']].drop_duplicates())

# %%
# prediction with light gbm
test_df['preds'] = lgbm(train_df, test_df, col, lgb_params)

# %%
test_df['preds'] = lgbm(new_train_df, new_test_df, col, lgb_params)

# %%
fig = px.scatter_mapbox(test_df,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg_bs",
                    lon="lngDeg_bs",
                    text='phoneName',

                    #Here, plotly detects color of series
                    color="preds",
                    labels="collectionName",

                    zoom=14.5,
                    center={"lat":37.334, "lon":-121.89},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()