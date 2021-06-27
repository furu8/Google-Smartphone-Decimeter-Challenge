import pandas as pd
import numpy as np

from models import Runner, ModelLGB

from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

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

def main():
    phone_name = 'Mi8'

    # pathはinterimではなくprocessedが正しいが動作用のため一旦これで
    train_df = pd.read_csv(f'../../data/processed/confirm/train/2021-06-25_20_{phone_name}.csv')
    # test_df = pd.read_csv(f'../../data/processed/confirm/test/2021-06-25_20_{phone_name}.csv') # 一旦無視
    baseline_train_df = pd.read_csv('../../data/raw/baseline_locations_train.csv')

    # score_df = pd.DataFrame()
    # score_df['lat_pred'] = baseline_train_df.loc[baseline_train_df['phoneName']=='Mi8', 'latDeg'].values
    # score_df['lng_pred'] = baseline_train_df.loc[baseline_train_df['phoneName']=='Mi8', 'lngDeg'].values
    # score_df['lat_truth'] = train_df['latDeg'].values
    # score_df['lng_truth'] = train_df['lngDeg'].values
    # score_df['dist'] = evaluate_lat_lng_dist(score_df)

    # dist50 = np.percentile(score_df['dist'], 50)
    # dist95 = np.percentile(score_df['dist'], 95)
    # print('-------')
    # print(dist50)
    # print(dist95)
    # print((dist50 + dist95) / 2)

    train_df['latDegBase'] = baseline_train_df.loc[baseline_train_df['phoneName']=='Mi8', 'latDeg'].values
    train_df['lngDegBase'] = baseline_train_df.loc[baseline_train_df['phoneName']=='Mi8', 'lngDeg'].values
    train_x = train_df.drop(['latDeg', 'lngDeg', 'phoneName', 'collectionName'], axis=1)
    train_y = train_df[['latDeg', 'lngDeg']]
    run_name = 'lgb'
    params = { 
        'max_depth' : 10,
        'num_leaves' : 300,
        'learning_rate' : 0.1,
        'n_estimators': 100,
        'objective': 'regression', # 目的 : 回帰  
        'metric': {'rmse'},        # 評価指標 : rsme(平均二乗誤差の平方根) 
        'verbose' : -1
    }

    runner = Runner(train_x, train_y, run_name, ModelLGB, ModelLGB, params)
    score_df_list = runner.run_train_cv()
    for score_df in score_df_list:
        eval_df = pd.DataFrame()
        dist50 = np.percentile(score_df['dist'], 50)
        dist95 = np.percentile(score_df['dist'], 95)
        print('-------')
        print(dist50)
        print(dist95)
        print((dist50 + dist95) / 2)
    


if __name__ == '__main__':
    main()