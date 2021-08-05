# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna.integration.lightgbm as lgb_o
# from models import Runner, ModelLGB
from sklearn.model_selection import KFold, TimeSeriesSplit
import matplotlib.pyplot as plt
import glob as gb
import re
import pyproj
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
bl_trn_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')
bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
display(bl_trn_df)
display(bl_tst_df)
# %%
train_pahts = gb.glob('../../data/interim/stacking/*.csv')

latlng_dict = {}
for train_path in train_pahts:
    key = re.split('/|.csv', train_path)[5]
    print(key)
    latlng_dict[key] = pd.read_csv(train_path)

# %%
# rename
for key in latlng_dict.keys():
    print(key)
    latlng_dict[key] = latlng_dict[key].rename(columns={'latDeg_pred':f'latDeg_{key}', 'lngDeg_pred':f'lngDeg_{key}'})
    display(latlng_dict[key])

# %%
df = pd.DataFrame()
for key in latlng_dict.keys():
    df = pd.concat([df, latlng_dict[key]], axis=1)
df = df.loc[:,~df.columns.duplicated()]
df
# %%
train_df = df[df['collectionName'].isin([
                        '2021-04-22-US-SJC-1',
                        '2021-04-26-US-SVL-1',
                        '2021-04-28-US-SJC-1',
                        '2021-04-29-US-SJC-2',
                        '2021-04-15-US-MTV-1',
                        '2021-04-28-US-MTV-1',
                        '2021-04-29-US-MTV-1',
                        '2021-03-10-US-SVL-1'])]
train_df.info()

# %%
test_df = df[df['collectionName'].isin([
                        '2021-04-21-US-MTV-1', 
                        '2021-04-28-US-MTV-2', 
                        '2021-04-29-US-MTV-2',
                        '2021-03-16-US-RWC-2', 
                        '2021-04-22-US-SJC-2', 
                        '2021-04-29-US-SJC-3'])]
test_df.info()


# %%
df_train = train_df.copy()
df_test = test_df.drop(['latDeg_gt', 'lngDeg_gt'], axis=1).reset_index(drop=True)
df_test


# %%
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

def train_cv(df_train, df_test, target, params):
    feature_names = df_train.drop(['latDeg_gt', 'lngDeg_gt', 'collectionName', 'phoneName'], axis=1).columns # gt除外

    kfold = KFold(n_splits=4, shuffle=True, random_state=2021)

    pred_valid = np.zeros((len(df_train),)) 
    pred_test = np.zeros((len(df_test),)) 
    scores, models = [], []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_valid = lgb.Dataset(X_val, Y_val)

        # model = lgb.LGBMRegressor(**params)
        # lgb_model = model.fit(X_train,
        #                       Y_train,
        #                       eval_names=['train', 'valid'],
        #                       eval_set=[(X_train, Y_train), (X_val, Y_val)],
        #                       verbose=0,
        #                       eval_metric=params['metric'],
        #                       early_stopping_rounds=params['early_stopping_rounds']
        #             )

        # パラメータ探索
        best_params, tuning_history = dict(), list()
        model = lgb_o.train(params, lgb_train, valid_sets=lgb_valid,
                            verbose_eval=0,
                            # best_params=best_params,
                            # tuning_history=tuning_history
                    )
        best_params = model.params
        print('Best Params:', best_params)
        print('Tuning history:', tuning_history)

        # 学習
        lgb_model = lgb_o.train(best_params, lgb_train, valid_sets=lgb_valid)

        # 予測
        # AttributeError: 'Booster' object has no attribute 'best_iteration_'
        pred_valid[val_idx] = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration)
        pred_test += lgb_model.predict(df_test[feature_names], num_iteration = lgb_model.best_iteration)
        # pred_valid[val_idx] = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration_)
        # pred_test += lgb_model.predict(df_test[feature_names], num_iteration = lgb_model.best_iteration_)

        scores.append(lgb_model.best_score_['valid']['l2'])
        # models.append(model)
        models.append(lgb_model)
    
    pred_test = pred_test / kfold.n_splits
    
    # if verbose_flag == True:
    print("Each Fold's MSE：{}, Average MSE：{:.4f}".format([np.round(v,2) for v in scores], np.mean(scores)))
    print("-"*60)
    
    return df_train, df_test, pred_valid, pred_test, models

# %%
params = {
    'metric':'mse',
    'objective':'regression',
    'seed':2021,
    'boosting_type':'gbdt',
    'early_stopping_rounds':10,
    'subsample':0.7,
    'feature_fraction':0.7,
    'bagging_fraction': 0.7,
    'reg_lambda': 10
    }

window_size = 30
bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
sub = pd.read_csv('../../data/submission/sample_submission.csv')
cn2pn_tst_df = bl_tst_df[['collectionName', 'phoneName']].drop_duplicates()

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
            '2021-04-29-US-MTV-2',
            '2021-03-16-US-RWC-2'
        ],
    # 'SVL': ['2021-04-26-US-SVL-2'],
}

for drived_id in cns_dict.keys():
    print(drived_id)
    if drived_id == 'SJC':
        df_trn, df_tst = extract_SJC(df_train, df_test)
    elif drived_id == 'MTV':
        df_trn, df_tst = extract_MTV(df_train, df_test)

    df_train_lat, df_test_lat, pred_valid_lat, pred_test_lat, models_lat = train_cv(df_trn, df_tst, 'latDeg_gt', params)
    df_train_lng, df_test_lng, pred_valid_lng, pred_test_lng, models_lng = train_cv(df_trn, df_tst, 'lngDeg_gt', params)

    val_compare_df = pd.DataFrame({'latDeg_gt':df_train_lat['latDeg_gt'].values, 'latDeg_pred':pred_valid_lat,
                                'lngDeg_gt':df_train_lat['lngDeg_gt'].values, 'lngDeg_pred':pred_valid_lng
                            })
    
    test_pred_df = pd.DataFrame({'latDeg':pred_test_lat, 'lngDeg':pred_test_lng})

    val_compare_df = pd.concat([val_compare_df, df_trn[['collectionName', 'phoneName']]], axis=1)
    display(val_compare_df)
    # 予測値にcollectionNameとphoneNameを結合
    test_pred_df = pd.concat([test_pred_df, df_tst[['collectionName', 'phoneName']]], axis=1)

    # IMU Prediction vs. GT
    val_compare_df['dist'] = calc_haversine(val_compare_df['latDeg_gt'], val_compare_df['lngDeg_gt'], 
                                    val_compare_df['latDeg_pred'], val_compare_df['lngDeg_pred'])
    # IMU预测vsGT（多collection）
    print('dist_50:',np.percentile(val_compare_df['dist'],50) )
    print('dist_95:',np.percentile(val_compare_df['dist'],95) )
    print('avg_dist_50_95:',(np.percentile(val_compare_df['dist'],50) + np.percentile(val_compare_df['dist'],95))/2)
    print('avg_dist:', val_compare_df['dist'].mean())

    display(test_pred_df)
    for cn in cns_dict[drived_id]:
        pns = cn2pn_tst_df.loc[cn2pn_tst_df['collectionName'] == cn, 'phoneName'].values
        for pn in pns:
            print(len(bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 3]))
            print(len(test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'latDeg']))
            # display(bl_tst_df[bl_tst_df['phone']==cn + '_' + pn])
            # display(bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:])
            bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 3] = test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'latDeg'].values
            bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 4] = test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'lngDeg'].values

# save
output = bl_tst_df[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].copy()
display(sub[sub['millisSinceGpsEpoch']!=output['millisSinceGpsEpoch']]) # 空だと良い
output.to_csv(f'../../data/submission/imu_many_lat_lng_deg_lgb3_rf2_rr_.csv', index=False)
# %%

"""
SJC
dist_50: 2.8937154350445953
dist_95: 9.766247142391551
avg_dist_50_95: 6.3299812887180735
avg_dist: 3.7743235268513273

MTV
dist_50: 2.347822238735025
dist_95: 8.8279213061084
avg_dist_50_95: 5.587871772421713
avg_dist: 3.096369958728955
"""