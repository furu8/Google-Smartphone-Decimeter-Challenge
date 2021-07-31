# %%
import pandas as pd
import numpy as np
import glob as gb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.model_selection import KFold
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

    if dataset_name == 'train':
        df_merge = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps"]].sort_values('millisSinceGpsEpoch'), acc_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
        df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), mag_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
        df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "latDeg_gt", "lngDeg_gt", "latDeg_bs", "lngDeg_bs", "heightAboveWgs84EllipsoidM", "speedMps", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyr_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
    elif dataset_name == 'test':
        df_merge = pd.merge_asof(df[["collectionName", "phoneName", "millisSinceGpsEpoch"]].sort_values('millisSinceGpsEpoch'), acc_df[["millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
        df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce"]].sort_values('millisSinceGpsEpoch'), mag_df[["millisSinceGpsEpoch", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')
        df_merge = pd.merge_asof(df_merge[["collectionName", "phoneName", "millisSinceGpsEpoch", "x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn"]].sort_values('millisSinceGpsEpoch'), gyr_df[["millisSinceGpsEpoch", "x_f_gyro", "y_f_gyro", "z_f_gyro"]].sort_values('millisSinceGpsEpoch'), on='millisSinceGpsEpoch', direction='nearest')

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

def make_test(sub, test_cname):
    test_df = pd.DataFrame()
    for cname in test_cname:
        phone_list = sub[sub['collectionName'] == cname]['phoneName'].drop_duplicates()
        for pname in phone_list:
            df = sub[(sub['collectionName'] == cname) & (sub['phoneName'] == pname)]
            # acc_df, gyr_df, mag_df = add_IMU('train', cname, pname)
            df = add_IMU(df, 'test', cname, pname)
            # df = merge_df(df, acc_df, gyr_df, mag_df)
            test_df = pd.concat([test_df, df])
    
    return test_df

# %%
def lgbm(train, test, col, lgb_params):
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(train[col], train['tag'])
    preds = model.predict(test[col])
    print('confusion matrix :  \n', confusion_matrix(preds, test['tag']))
    print('accuracy score : ', accuracy_score(preds, test['tag']))
    return preds

# %%
def train_cv(df_train, df_test, col, params):
    kfold = KFold(n_splits=3, shuffle=True, random_state=2021)

    pred_valid = np.zeros((len(df_train),)) 
    pred_test = np.zeros((len(df_test),)) 
    models = []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train['tag'])):
        X_train = df_train.iloc[trn_idx][col]
        Y_train = df_train.iloc[trn_idx]['tag']
        X_val = df_train.iloc[val_idx][col]
        Y_val = df_train.iloc[val_idx]['tag']

        # lgb_train = lgb.Dataset(X_train, Y_train)
        # lgb_valid = lgb.Dataset(X_val, Y_val)

        model = lgb.LGBMClassifier(**params)
        lgb_model = model.fit(X_train, 
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                            #   eval_metric=params['metric'],
                              early_stopping_rounds=params['early_stopping_rounds'])

        # パラメータ探索
        # best_params, tuning_history = dict(), list()
        # lgb_o.train(params, lgb_train, valid_sets=lgb_valid,
        #                     verbose_eval=0,
        #                     # best_params=best_params,
        #                     # tuning_history=tuning_history
        #             )
        # best_params = lgb_o.params
        # print('Best Params:', best_params)
        # print('Tuning history:', tuning_history)

        # 学習
        # lgb_model = lgb_o.train(best_params, lgb_train, valid_sets=lgb_valid)

        # 予測
        pred_valid[val_idx] = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration_)
        pred_test += lgb_model.predict(df_test[col], num_iteration = lgb_model.best_iteration_)

        # models.append(model)
        models.append(lgb_model)
    
    pred_test = pred_test / kfold.n_splits
    
    
    return df_train, df_test, pred_valid, pred_test, models

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
def print_importances(models, cols):
    for i, model in enumerate(models):
        print('fold ', i+1)
        print(len(model.feature_importances_))
        importance = pd.DataFrame(model.feature_importances_, index=cols, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        display(importance.head(20))

# %%
train_cname = [
    '2021-04-22-US-SJC-1',
    '2021-04-26-US-SVL-1',
    '2021-04-28-US-SJC-1',
    '2021-04-29-US-SJC-2',
    '2021-04-15-US-MTV-1',
    '2021-04-29-US-MTV-1',
    '2021-03-10-US-SVL-1', 
    '2021-04-28-US-MTV-1',
]
test_cname = [
    '2021-04-02-US-SJC-1',
    '2021-04-22-US-SJC-2',
    '2021-04-29-US-SJC-3',
    '2021-03-16-US-MTV-2',
    '2021-04-08-US-MTV-1',
    '2021-04-21-US-MTV-1',
    '2021-04-28-US-MTV-2',
    '2021-04-29-US-MTV-2',
    '2021-04-26-US-SVL-2',
    '2021-03-16-US-RWC-2',
    '2021-03-25-US-PAO-1'
]


tag_v = 0.5
col = ["x_f_acce", "y_f_acce", "z_f_acce", "x_f_magn", "y_f_magn", "z_f_magn", "x_f_gyro", "y_f_gyro", "z_f_gyro"]

# parameter
lgb_params = {
    'num_leaves': 90,
    'n_estimators': 125,
    'early_stopping_rounds':10,
}

sub = pd.read_csv('../../data/submission/sample_submission.csv')
sub
# %%
sub = pd.concat([sub, sub['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
sub
# %%
%%time
# make train&test
train_df = make_train(train_cname, tag_v)
# test_df = make_train(test_cname, tag_v)
test_df = make_test(sub, test_cname)

# %%
# save
train_df.to_csv('../../data/interim/train/train_org_moving_or_not_4interpolate.csv', index=False)
test_df.to_csv('../../data/interim/test/test_org_moving_or_not_4interpolate.csv', index=False)

# %%
display(train_df)
display(test_df)

# %%
display(train_df[['collectionName', 'phoneName']].drop_duplicates())
display(test_df[['collectionName', 'phoneName']].drop_duplicates())

# # %%
# sjc_pi4_trn = train_df[(train_df['collectionName']=='2021-04-29-US-SJC-2') & (train_df['phoneName']=='Pixel4')]
# sjc_sam_trn = train_df[(train_df['collectionName']=='2021-04-29-US-SJC-2') & (train_df['phoneName']=='SamsungS20Ultra')]
# svl_pi4xl_trn = train_df[(train_df['collectionName']=='2021-03-10-US-SVL-1') & (train_df['phoneName']=='Pixel4XL')]
# svl_sam_trn = train_df[(train_df['collectionName']=='2021-03-10-US-SVL-1') & (train_df['phoneName']=='SamsungS20Ultra')]
# sjc_pi4_tst = test_df[(test_df['collectionName']=='2021-04-28-US-SJC-1') & (test_df['phoneName']=='Pixel4')]
# sjc_sam_tst = test_df[(test_df['collectionName']=='2021-04-28-US-SJC-1') & (test_df['phoneName']=='SamsungS20Ultra')]

# # %%
# new_train_df = pd.concat([sjc_sam_trn, sjc_pi4_trn])
# new_train_df = pd.concat([new_train_df, svl_sam_trn])
# new_train_df = pd.concat([new_train_df, svl_pi4xl_trn])
# new_train_df

# # %%
# new_test_df = pd.concat([sjc_sam_tst, sjc_pi4_tst])
# new_test_df

# # %%
# display(new_train_df[['collectionName', 'phoneName']].drop_duplicates())
# display(new_test_df[['collectionName', 'phoneName']].drop_duplicates())

# %%
# prediction with light gbm
# test_df['preds'] = lgbm(train_df, test_df, col, lgb_params)

# # %%
# test_df['preds'] = lgbm(new_train_df, new_test_df, col, lgb_params)

# %%
%%time
# cv
df_train, df_test, pred_valid, pred_test, models = train_cv(train_df, test_df, col, lgb_params)
val_compare_df = pd.DataFrame({'tag_gt':df_train['tag'].values, 'tag_pred':pred_valid,})
val_compare_df

# %%
print('confusion matrix :  \n', confusion_matrix(val_compare_df['tag_pred'], val_compare_df['tag_gt']))
print('accuracy score : ', accuracy_score(val_compare_df['tag_pred'], val_compare_df['tag_gt']))


"""
train:
collectionName	phoneName
2021-04-22-US-SJC-1	Pixel4
2021-04-22-US-SJC-1	SamsungS20Ultra
2021-04-26-US-SVL-1	Pixel5
2021-04-26-US-SVL-1	Mi8
2021-04-28-US-SJC-1	Pixel4
2021-04-28-US-SJC-1	SamsungS20Ultra
2021-04-29-US-SJC-2	Pixel4
2021-04-29-US-SJC-2	SamsungS20Ultra
2021-04-15-US-MTV-1	Pixel5
2021-04-15-US-MTV-1	Pixel4
2021-04-15-US-MTV-1	Pixel4Modded
2021-04-15-US-MTV-1	SamsungS20Ultra
2021-04-29-US-MTV-1	Pixel5
2021-04-29-US-MTV-1	Pixel4
2021-04-29-US-MTV-1	SamsungS20Ultra

confusion matrix :  
 [[18459   876]
 [  365  8645]]
accuracy score :  0.9562180278708767
"""

"""
train:
collectionName	phoneName
2021-04-22-US-SJC-1	Pixel4
2021-04-22-US-SJC-1	SamsungS20Ultra
2021-04-26-US-SVL-1	Pixel5
2021-04-26-US-SVL-1	Mi8
2021-04-28-US-SJC-1	Pixel4
2021-04-28-US-SJC-1	SamsungS20Ultra
2021-04-29-US-SJC-2	Pixel4
2021-04-29-US-SJC-2	SamsungS20Ultra
2021-04-15-US-MTV-1	Pixel5
2021-04-15-US-MTV-1	Pixel4
2021-04-15-US-MTV-1	Pixel4Modded
2021-04-15-US-MTV-1	SamsungS20Ultra
2021-04-29-US-MTV-1	Pixel5
2021-04-29-US-MTV-1	Pixel4
2021-04-29-US-MTV-1	SamsungS20Ultra
2021-04-28-US-SJC-1	Pixel4
2021-04-28-US-SJC-1	SamsungS20Ultra
2021-03-10-US-SVL-1	Pixel4XL
2021-03-10-US-SVL-1	SamsungS20Ultra
2021-04-28-US-MTV-1	Pixel5
2021-04-28-US-MTV-1	Pixel4
2021-04-28-US-MTV-1	SamsungS20Ultra

confusion matrix :  
 [[18459   876]
 [  365  8645]]
accuracy score :  0.9562180278708767
"""

# %%
# feature importanceを表示
print_importances(models, col)

# %%
bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
bl_tst_df

# %%
new_test_df = pd.merge_asof(
                    test_df.sort_values('millisSinceGpsEpoch'),
                    bl_tst_df[['millisSinceGpsEpoch', 'phoneName', 'collectionName', 'latDeg', 'lngDeg']].sort_values('millisSinceGpsEpoch'),
                    on='millisSinceGpsEpoch',
                    by=['collectionName', 'phoneName'],
                    direction='nearest'
)
new_test_df

# %%
new_test_df['tag_pred'] = pred_test
new_test_df
# %%
fig = px.scatter_mapbox(new_test_df,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg",
                    lon="lngDeg",
                    text='phoneName',

                    #Here, plotly detects color of series
                    color="tag_pred",
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
new_test_df['tag_pred'].hist()