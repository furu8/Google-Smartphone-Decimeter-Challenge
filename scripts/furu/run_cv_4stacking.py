# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_o
from models import Runner, ModelLGB, ModelRF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, TimeSeriesSplit
import matplotlib.pyplot as plt
import glob as gb
import pyproj
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)


def extract_SJC(train_df, test_df):
    # extract
    train_df = train_df[
                    (train_df['collectionName']=='2021-04-22-US-SJC-1') 
                    | (train_df['collectionName']=='2021-04-28-US-SJC-1')
                    | (train_df['collectionName']=='2021-04-29-US-SJC-2')
                ]
    test_df = test_df[
                    # (test_df['collectionName']=='2021-04-02-US-SJC-1') # これだけ場所が違う
                    (test_df['collectionName']=='2021-04-22-US-SJC-2')
                    | (test_df['collectionName']=='2021-04-29-US-SJC-3')
                ]

    # drop
    # train_df = train_df.drop(['collectionName', 'phoneName'], axis=1)
    # test_df = test_df.drop(['collectionName', 'phoneName'], axis=1)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def extract_MTV(train_df, test_df):
    # extract
    train_df = train_df[
                    # (train_df['collectionName']=='2021-04-15-US-MTV-1')
                    (train_df['collectionName']=='2021-04-28-US-MTV-1')
                    | (train_df['collectionName']=='2021-04-29-US-MTV-1')
                ]
    test_df = test_df[
                    # (test_df['collectionName']=='2021-03-16-US-MTV-2')
                    # | (test_df['collectionName']=='2021-04-08-US-MTV-1')
                    (test_df['collectionName']=='2021-04-21-US-MTV-1')
                    | (test_df['collectionName']=='2021-04-28-US-MTV-2')
                    | (test_df['collectionName']=='2021-04-29-US-MTV-2')
                    | (test_df['collectionName']=='2021-03-16-US-RWC-2')
                ]

    # drop
    # train_df = train_df.drop(['collectionName', 'phoneName'], axis=1)
    # test_df = test_df.drop(['collectionName', 'phoneName'], axis=1)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def extract_SVL(train_df, test_df):
    # extract
    train_df = train_df[
                    (train_df['collectionName']=='2021-03-10-US-SVL-1')
                ]
    test_df = test_df[
                    (test_df['collectionName']=='2021-04-26-US-SVL-2')
                ]

    # drop
    # train_df = train_df.drop(['collectionName', 'phoneName'], axis=1)
    # test_df = test_df.drop(['collectionName', 'phoneName'], axis=1)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def evaluate_lat_lng_dist(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Radius of earth in kilometers is 6367 or 6371
    RADIUS = 6371000
    # RADIUS = 6367000
    
    dist_list = []

    for i in range(len(df)):
        lat_truth = df.loc[i, 'lat_truth']
        lng_truth = df.loc[i, 'lng_truth']
        lat_pred = df.loc[i, 'lat_pred']
        lng_pred = df.loc[i, 'lng_pred']
        # convert decimal degrees to radians 
        lng_truth, lat_truth, lng_pred, lat_pred = map(np.deg2rad, [lng_truth, lat_truth, lng_pred, lat_pred])
        # haversine formula 
        dlng = lng_pred - lng_truth 
        dlat = lat_pred - lat_truth 
        a = np.sin(dlat/2)**2 + np.cos(lat_truth) * np.cos(lat_pred) * np.sin(dlng/2)**2
        dist = 2 * RADIUS * np.arcsin(np.sqrt(a))
        dist_list.append(dist)

    return dist_list

def train_cv(df_train, df_test, tgt_axis, run_name, stacking_model, params):
    feature_names = df_train.drop(['Xgt', 'Ygt', 'Zgt'], axis=1).columns # gt除外
    target = '{}gt'.format(tgt_axis)

    kfold = KFold(n_splits=4, shuffle=True, random_state=2021)

    pred_valid = np.zeros((len(df_train),)) 
    pred_test = np.zeros((len(df_test),)) 
    scores, models = [], []
    
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        model = stacking_model(f'{run_name}_{fold_id+1}', params)
        model.train(X_train, Y_train, X_val, Y_val)

        # lgb_train = lgb.Dataset(X_train, Y_train)
        # lgb_valid = lgb.Dataset(X_val, Y_val)

        # パラメータ探索
        # best_params, tuning_history = dict(), list()
        # model = lgb_o.train(params, lgb_train, valid_sets=lgb_valid,
        #                     verbose_eval=0,
        #                     # best_params=best_params,
        #                     # tuning_history=tuning_history
        #             )
        # best_params = model.params
        # print('Best Params:', best_params)
        # print('Tuning history:', tuning_history)

        # 学習
        # lgb_model = lgb_o.train(best_params, lgb_train, valid_sets=lgb_valid)

        # 予測
        pred_valid[val_idx] = model.predict(X_val)
        # pred_valids.append(pred_valid)
        # val_idxes.append(val_idx)

        pred_test += model.predict(df_test[feature_names])

        scores.append(mean_squared_error(Y_val, pred_valid[val_idx]))
        models.append(model.model)
    
    # val_idxes = np.concatenate(val_idxes)
    # pred_valids = np.concatenate(pred_valids, axis=0)
    # pred_train = pred_valid[np.argsort(val_idxes)]
    
    pred_test = pred_test / kfold.n_splits
    
    # if verbose_flag == True:
    print("Each Fold's MSE：{}, Average MSE：{:.4f}".format([np.round(v,2) for v in scores], np.mean(scores)))
    print("-"*60)

    # df_train[f'pred_{tgt_axis}']
    
    return df_train, df_test, pred_valid, pred_test, models

def ECEF_to_WGS84(x, y, z):
    transformer = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},)
    lon, lat, alt = transformer.transform(x,y,z,radians=False)
    return lon, lat, alt

def print_importances(models, cols):
    for i, model in enumerate(models):
        print('fold ', i+1)
        print(len(model.feature_importances_))
        importance = pd.DataFrame(model.feature_importances_, index=cols, columns=['importance'])
        importance = importance.sort_values('importance', ascending=False)
        display(importance.head(20))

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

def run_learing(drived_id, x_trn_df, x_tst_df, y_trn_df, y_tst_df, z_trn_df, z_tst_df, run_name, model, params):
    print(drived_id)
    if drived_id == 'SJC':
        x_train_df, x_test_df = extract_SJC(x_trn_df, x_tst_df)
        y_train_df, y_test_df = extract_SJC(y_trn_df, y_tst_df)
        z_train_df, z_test_df = extract_SJC(z_trn_df, z_tst_df)
    elif drived_id == 'MTV':
        x_train_df, x_test_df = extract_MTV(x_trn_df, x_tst_df)
        y_train_df, y_test_df = extract_MTV(y_trn_df, y_tst_df)
        z_train_df, z_test_df = extract_MTV(z_trn_df, z_tst_df)
    elif drived_id == 'SVL':
        x_train_df, x_test_df = extract_SVL(x_trn_df, x_tst_df)
        y_train_df, y_test_df = extract_SVL(y_trn_df, y_tst_df)
        z_train_df, z_test_df = extract_SVL(z_trn_df, z_tst_df)
    
    # drop
    x_train_df_droped = x_train_df.drop(['collectionName', 'phoneName'], axis=1)
    x_test_df_droped = x_test_df.drop(['collectionName', 'phoneName'], axis=1)
    y_train_df_droped = y_train_df.drop(['collectionName', 'phoneName'], axis=1)
    y_test_df_droped = y_test_df.drop(['collectionName', 'phoneName'], axis=1)
    z_train_df_droped = z_train_df.drop(['collectionName', 'phoneName'], axis=1)
    z_test_df_droped = z_test_df.drop(['collectionName', 'phoneName'], axis=1)

    df_train_x, df_test_x, pred_valid_x, pred_test_x, models_x = train_cv(x_train_df_droped, x_test_df_droped, 'X', run_name, model, params)
    df_train_y, df_test_y, pred_valid_y, pred_test_y, models_y = train_cv(y_train_df_droped, y_test_df_droped, 'Y', run_name, model, params)
    df_train_z, df_test_z, pred_valid_z, pred_test_z, models_z = train_cv(z_train_df_droped, z_test_df_droped, 'Z', run_name, model, params)

    val_compare_df = pd.DataFrame({'Xgt':df_train_x['Xgt'].values, 'Xpred':pred_valid_x,
                            'Ygt':df_train_y['Ygt'].values, 'Ypred':pred_valid_y,
                                'Zgt':df_train_z['Zgt'].values, 'Zpred':pred_valid_z
                            })

    # feature importanceを表示
    # print_importances(models_x, df_test_x.columns)
    # print_importances(models_y, df_test_y.columns)
    # print_importances(models_z, df_test_z.columns)

    # xyz -> lng, lat
    lng_gt, lat_gt, _ = ECEF_to_WGS84(val_compare_df['Xgt'].values,val_compare_df['Ygt'].values,val_compare_df['Zgt'].values)
    lng_pred, lat_pred, _ = ECEF_to_WGS84(val_compare_df['Xpred'].values,val_compare_df['Ypred'].values,val_compare_df['Zpred'].values)
    lng_test_pred, lat_test_pred, _ = ECEF_to_WGS84(pred_test_x, pred_test_y, pred_test_z)

    val_compare_df['latDeg_gt'] = lat_gt
    val_compare_df['lngDeg_gt'] = lng_gt
    val_compare_df['latDeg_pred'] = lat_pred
    val_compare_df['lngDeg_pred'] = lng_pred
    test_pred_df = pd.DataFrame({'latDeg':lat_test_pred, 'lngDeg':lng_test_pred})
    # display(test_pred_df)

    val_compare_df = pd.concat([val_compare_df, x_train_df[['collectionName', 'phoneName']]], axis=1)
    # 予測値にcollectionNameとphoneNameを結合
    test_pred_df = pd.concat([test_pred_df, x_test_df[['collectionName', 'phoneName']]], axis=1)
    # display(test_pred_df)
    
    # plot
    val_compare_df[['Xgt', 'Xpred']].plot(figsize=(16,8))
    plt.show()
    val_compare_df[['Ygt', 'Ypred']].plot(figsize=(16,8))
    plt.show()
    val_compare_df[['Zgt', 'Zpred']].plot(figsize=(16,8))
    plt.show()

    # IMU Prediction vs. GT
    val_compare_df['dist'] = calc_haversine(val_compare_df['latDeg_gt'], val_compare_df['lngDeg_gt'], 
                                    val_compare_df['latDeg_pred'], val_compare_df['lngDeg_pred'])
    # IMU预测vsGT（多collection）
    print('dist_50:',np.percentile(val_compare_df['dist'],50) )
    print('dist_95:',np.percentile(val_compare_df['dist'],95) )
    print('avg_dist_50_95:',(np.percentile(val_compare_df['dist'],50) + np.percentile(val_compare_df['dist'],95))/2)
    print('avg_dist:', val_compare_df['dist'].mean())

    val_compare_df = val_compare_df[['latDeg_gt', 'lngDeg_gt', 'latDeg_pred', 'lngDeg_pred', 'collectionName', 'phoneName']].copy()

    stacking_df = pd.concat([val_compare_df, test_pred_df[['latDeg', 'lngDeg', 'collectionName', 'phoneName']].rename(columns={'latDeg': 'latDeg_pred', 'lngDeg': 'lngDeg_pred'})], axis=0)

    return test_pred_df, stacking_df.reset_index(drop=True)

# %%
%%time
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
            '2021-04-29-US-MTV-2',
            '2021-03-16-US-RWC-2'
        ],
    # 'SVL': ['2021-04-26-US-SVL-2'],
}
# '2021-03-25-US-PAO-1'
window_size = 30

stacking_dfs = pd.DataFrame()
bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
sub = pd.read_csv('../../data/submission/sample_submission.csv')
cn2pn_tst_df = bl_tst_df[['collectionName', 'phoneName']].drop_duplicates()

# display(train_df)
# print(test_df.columns)

# ここをいじる
# run_name = 'lgbm' # 名前変えてね
# params = {
#     'max_depth': 10,
#     'num_leaves': 1024,
#     'metric':'mse',
#     'objective':'regression',
#     'seed':2021,
#     'boosting_type':'gbdt',
#     'early_stopping_rounds':10,
#     'subsample':0.7,
#     'feature_fraction':0.7,
#     'bagging_fraction': 0.7,
#     'reg_lambda': 10
# }

run_name = 'rfm'
params = {
    'max_depth': 10,
    'num_leaves': 1024,
    'n_estimators': 100,
    'random_state': 2021,
}

x_trn_df = pd.read_csv(f'../../data/processed/train/imu_x_many_lat_lng_deg.csv')
x_tst_df = pd.read_csv(f'../../data/processed/test/imu_x_many_lat_lng_deg.csv')
y_trn_df = pd.read_csv(f'../../data/processed/train/imu_y_many_lat_lng_deg.csv')
y_tst_df = pd.read_csv(f'../../data/processed/test/imu_y_many_lat_lng_deg.csv')
z_trn_df = pd.read_csv(f'../../data/processed/train/imu_z_many_lat_lng_deg.csv')
z_tst_df = pd.read_csv(f'../../data/processed/test/imu_z_many_lat_lng_deg.csv')

for drived_id in cns_dict.keys():
    test_pred_df, stacking_df = run_learing(drived_id, x_trn_df, x_tst_df, y_trn_df, y_tst_df, z_trn_df, z_tst_df, run_name, ModelLGB, params)
    # test_pred_df, stacking_df = run_learing(drived_id, x_trn_df, x_tst_df, y_trn_df, y_tst_df, z_trn_df, z_tst_df, run_name, ModelRF, params)
    stacking_dfs = pd.concat([stacking_dfs, stacking_df], axis=0) 

display(stacking_dfs)

# lgbm
"""maxdepth:-1, n_estimators:100
SJC
dist_50: 3.532140410316554
dist_95: 11.024273029511706
avg_dist_50_95: 7.27820671991413
avg_dist: 4.50314864523743

MTV
dist_50: 2.5158099830554246
dist_95: 8.057531182140774
avg_dist_50_95: 5.2866705825980995
avg_dist: 3.1502449611507966
"""

"""maxdepth:10, n_estimators:100
SJC
dist_50: 3.576438316207736
dist_95: 11.059847026820272
avg_dist_50_95: 7.318142671514003
avg_dist: 4.523908368641834

MTV
dist_50: 2.5111875475969567
dist_95: 8.060663313276235
avg_dist_50_95: 5.2859254304365955
avg_dist: 3.1479418943562054
"""

"""maxdepth:30, n_estimators:100
SJC
dist_50: 3.532140410316554
dist_95: 11.024273029511704
avg_dist_50_95: 7.278206719914129
avg_dist: 4.503148645236647

MTV
dist_50: 2.5158099830554246
dist_95: 8.057531181616632
avg_dist_50_95: 5.286670582336028
avg_dist: 3.150244961153496
"""

"""max_depth:5, num_leaves:31, n_estimators:100
dist_50: 4.084558018493
dist_95: 12.9402931036267
avg_dist_50_95: 8.51242556105985
avg_dist: 5.211693035169141

dist_50: 2.584113011551043
dist_95: 8.161943133386611
avg_dist_50_95: 5.373028072468827
avg_dist: 3.239827717720705
"""

# rfm
"""max_depth:30 n_estimators:100
SJC
dist_50: 2.7086529980728122
dist_95: 11.325458192655608
avg_dist_50_95: 7.01705559536421
avg_dist: 3.953079105888895

MTV
dist_50: 1.313743888356429
dist_95: 4.68743609411494
avg_dist_50_95: 3.0005899912356844
avg_dist: 1.7378300969927243
"""

"""max_depth:10, n_estimators:100
SJC
dist_50: 3.166285673774718
dist_95: 12.382431128206115
avg_dist_50_95: 7.774358400990416
avg_dist: 4.482838921352679

MTV
dist_50: 1.446185834653453
dist_95: 4.8529391108088245
avg_dist_50_95: 3.1495624727311387
avg_dist: 1.8870523784103994
"""
# %%
# save
stacking_dfs.to_csv(f'../../data/interim/stacking/imu_{run_name}_maxdepth30_n100.csv', index=False)
# %%
