# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
# from models import Runner, ModelLGB
from sklearn.model_selection import KFold, TimeSeriesSplit
import matplotlib.pyplot as plt
import pyproj
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)


def load_dfs(drived_id):
    x_train_df = pd.read_csv(f'../../data/processed/train/imu_x_many_lat_lng_deg.csv')
    x_test_df = pd.read_csv(f'../../data/processed/test/imu_x_many_lat_lng_deg.csv')
    y_train_df = pd.read_csv(f'../../data/processed/train/imu_y_many_lat_lng_deg.csv')
    y_test_df = pd.read_csv(f'../../data/processed/test/imu_y_many_lat_lng_deg.csv')
    z_train_df = pd.read_csv(f'../../data/processed/train/imu_z_many_lat_lng_deg.csv')
    z_test_df = pd.read_csv(f'../../data/processed/test/imu_z_many_lat_lng_deg.csv')

    if drived_id == 'SJC':
        x_train_df, x_test_df = extract_SJC(x_train_df, x_test_df)
        y_train_df, y_test_df = extract_SJC(y_train_df, y_test_df)
        z_train_df, z_test_df = extract_SJC(z_train_df, z_test_df)
    elif drived_id == 'MTV':
        x_train_df, x_test_df = extract_MTV(x_train_df, x_test_df)
        y_train_df, y_test_df = extract_MTV(y_train_df, y_test_df)
        z_train_df, z_test_df = extract_MTV(z_train_df, z_test_df)
    elif drived_id == 'SVL':
        x_train_df, x_test_df = extract_SVL(x_train_df, x_test_df)
        y_train_df, y_test_df = extract_SVL(y_train_df, y_test_df)
        z_train_df, z_test_df = extract_SVL(z_train_df, z_test_df)

    return x_train_df, x_test_df, y_train_df, y_test_df, z_train_df, z_test_df

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

def train_cv(df_train, df_test, tgt_axis, params):
    feature_names = df_train.drop(['Xgt', 'Ygt', 'Zgt'], axis=1).columns # gt除外
    target = '{}gt'.format(tgt_axis)

    kfold = KFold(n_splits=3, shuffle=True, random_state=params['seed'])

    pred_valid = np.zeros((len(df_train),)) 
    pred_test = np.zeros((len(df_test),)) 
    scores, models = [], []
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train, df_train[target])):
        X_train = df_train.iloc[trn_idx][feature_names]
        Y_train = df_train.iloc[trn_idx][target]
        X_val = df_train.iloc[val_idx][feature_names]
        Y_val = df_train.iloc[val_idx][target]

        model = lgb.LGBMRegressor(**params)
        lgb_model = model.fit(X_train, 
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=0,
                              eval_metric=params['metric'],
                              early_stopping_rounds=params['early_stopping_rounds']
                    )

        pred_valid[val_idx] = lgb_model.predict(X_val, num_iteration = lgb_model.best_iteration_)
        pred_test += lgb_model.predict(df_test[feature_names], num_iteration = lgb_model.best_iteration_)

        scores.append(lgb_model.best_score_['valid']['l2'])
        models.append(model)
    
    pred_test = pred_test / kfold.n_splits
    
    # if verbose_flag == True:
    print("Each Fold's MSE：{}, Average MSE：{:.4f}".format([np.round(v,2) for v in scores], np.mean(scores)))
    print("-"*60)
    
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

# %%
%%time
def main():
    window_size = 30
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
        'SVL': ['2021-04-26-US-SVL-2'],
    }
    # '2021-03-25-US-PAO-1'

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

    bl_tst_df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
    cn2pn_tst_df = bl_tst_df[['collectionName', 'phoneName']].drop_duplicates()

    x_trn_df = pd.read_csv(f'../../data/processed/train/imu_x_aga_pred_phone.csv')
    x_tst_df = pd.read_csv(f'../../data/processed/test/imu_x_aga_pred_phone.csv')
    y_trn_df = pd.read_csv(f'../../data/processed/train/imu_x_aga_pred_phone.csv')
    y_tst_df = pd.read_csv(f'../../data/processed/test/imu_x_aga_pred_phone.csv')
    z_trn_df = pd.read_csv(f'../../data/processed/train/imu_x_aga_pred_phone.csv')
    z_tst_df = pd.read_csv(f'../../data/processed/test/imu_x_aga_pred_phone.csv')

    for drived_id in cns_dict.keys():
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
        display(x_train_df.shape)
        # display(x_test_df)

        # drop
        x_train_df_droped = x_train_df.drop(['collectionName', 'phoneName'], axis=1)
        x_test_df_droped = x_test_df.drop(['collectionName', 'phoneName'], axis=1)
        y_train_df_droped = y_train_df.drop(['collectionName', 'phoneName'], axis=1)
        y_test_df_droped = y_test_df.drop(['collectionName', 'phoneName'], axis=1)
        z_train_df_droped = z_train_df.drop(['collectionName', 'phoneName'], axis=1)
        z_test_df_droped = z_test_df.drop(['collectionName', 'phoneName'], axis=1)

        print(x_train_df_droped.shape)

        df_train_x, df_test_x, pred_valid_x, pred_test_x, models_x = train_cv(x_train_df_droped, x_test_df_droped, 'X', params)
        df_train_y, df_test_y, pred_valid_y, pred_test_y, models_y = train_cv(y_train_df_droped, y_test_df_droped, 'Y', params)
        df_train_z, df_test_z, pred_valid_z, pred_test_z, models_z = train_cv(z_train_df_droped, z_test_df_droped, 'Z', params)

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
        display(test_pred_df)
        # 予測値にcollectionNameとphoneNameを結合
        test_pred_df = pd.concat([test_pred_df, x_test_df[['collectionName', 'phoneName']]], axis=1)
        display(test_pred_df)
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

        # subに代入
        print(cns_dict[drived_id])
        for cn in cns_dict[drived_id]:
            pns = cn2pn_tst_df.loc[cn2pn_tst_df['collectionName'] == cn, 'phoneName'].values
            for pn in pns:
                print(cn, pn)
                print(test_pred_df['collectionName'].unique())
                print(test_pred_df['phoneName'].unique())
                print(len(bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 3]))
                print(len(test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'latDeg']))
                bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 3] = test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'latDeg'].values
                bl_tst_df.iloc[bl_tst_df[bl_tst_df['phone']==cn + '_' + pn].index[window_size-1:], 4] = test_pred_df.loc[(test_pred_df['collectionName']==cn) & (test_pred_df['phoneName']==pn), 'lngDeg'].values

    # save
    # bl_tst_df = bl_tst_df[['phone', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']]
    # bl_tst_df.to_csv(f'../../data/submission/imu_many_lat_lng_deg.csv', index=False)



"""imu_many_lat_lng_deg_4SJC
dist_50: 4.994871467980026
dist_95: 17.410167568075536
avg_dist_50_95: 11.202519518027781
avg_dist: 6.748650959462235
"""

"""imu_many_lat_lng_deg
dist_50: 5.8130661278406
dist_95: 27.36315219756584
avg_dist_50_95: 16.58810916270322
avg_dist: 9.004280045960275
"""



if __name__ == '__main__':
    main()
# %%
# 可視化関連
def visualize_trafic(df, center, zoom=15):
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
    
def visualize_collection(df):
    target_df = df
    lat_center = target_df['latDeg'].mean()
    lng_center = target_df['lngDeg'].mean()
    center = {"lat":lat_center, "lon":lng_center}
    
    visualize_trafic(target_df, center)
# %%
cn = '2021-03-16-US-MTV-2',
bl = pd.read_csv('../../data/raw/baseline_locations_test.csv')
# plt.scatter(bl_tst_df.loc[bl['collectionName']==cn, 'lngDeg'], bl_tst_df.loc[bl['collectionName']==cn, 'latDeg'])
# plt.scatter(bl.loc[bl['collectionName']==cn, 'lngDeg'], bl.loc[bl['collectionName']==cn, 'latDeg'], s=0.5)
# visualize_collection(bl[bl['collectionName'].isin(['2021-03-16-US-MTV-2', 
#                                             '2021-04-08-US-MTV-1', 
#                                             '2021-04-21-US-MTV-1', 
#                                             '2021-04-28-US-MTV-2', 
#                                             '2021-04-29-US-MTV-2'])])

# %%
# bl_tst
visualize_collection(bl_tst_df[bl_tst_df['collectionName'].isin(['2021-03-16-US-MTV-2', 
                                            '2021-04-08-US-MTV-1', 
                                            '2021-04-21-US-MTV-1', 
                                            '2021-04-28-US-MTV-2', 
                                            '2021-04-29-US-MTV-2'])])

# %%
# bl test sjc
visualize_collection(bl[bl['collectionName'].isin([
                                                # '2021-04-02-US-SJC-1',
                                                '2021-04-22-US-SJC-2',
                                                '2021-04-29-US-SJC-3',])])


# %%
# bl test mtv
visualize_collection(bl[bl['collectionName'].isin([
                                            # '2021-03-16-US-MTV-2', # 左の長い 
                                            # '2021-04-08-US-MTV-1', # 上のリボン
                                            '2021-04-21-US-MTV-1', 
                                            '2021-04-28-US-MTV-2', 
                                            '2021-04-29-US-MTV-2'
                                            ]))

# %%
# bl test svl
visualize_collection(bl[bl['collectionName'].isin(['2021-04-26-US-SVL-2'])])
# %%
# bl train sjc
bltr = pd.read_csv('../../data/raw/baseline_locations_train.csv')
visualize_collection(bltr[bltr['collectionName'].isin([
                                            '2021-04-22-US-SJC-1',
                                            '2021-04-28-US-SJC-1',
                                            '2021-04-29-US-SJC-2',
                                            ])]) 

# %%
# bl train mtv
visualize_collection(bltr[bltr['collectionName'].isin([
                                            # '2021-04-15-US-MTV-1', # 上側の複雑
                                            '2021-04-28-US-MTV-1',
                                            '2021-04-29-US-MTV-1',
                                            ])])             

# %%
# bl train svl
visualize_collection(bltr[bltr['collectionName'].isin([
                                        '2021-03-10-US-SVL-1'
                                            ])])
# %%
# train
'2021-04-22-US-SJC-1',
'2021-04-28-US-SJC-1',
'2021-04-29-US-SJC-2',
'2021-04-15-US-MTV-1',
'2021-04-28-US-MTV-1',
'2021-04-29-US-MTV-1',

# test
'2021-03-16-US-MTV-2', 
'2021-04-08-US-MTV-1', 
'2021-04-21-US-MTV-1', 
'2021-04-28-US-MTV-2', 
'2021-04-29-US-MTV-2'
'2021-04-02-US-SJC-1',
'2021-04-22-US-SJC-2',
'2021-04-29-US-SJC-3',