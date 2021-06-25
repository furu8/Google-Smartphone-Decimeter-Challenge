import pandas as pd
import numpy as np

from models import Runner, ModelLGB

from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

def main():
    phone_name = 'Mi8'

    # pathはinterimではなくprocessedが正しいが動作用のため一旦これで
    train_df = pd.read_csv(f'../../data/processed/confirm/train/2021-06-25_20_{phone_name}.csv')
    # test_df = pd.read_csv(f'../../data/processed/confirm/test/2021-06-25_20_{phone_name}.csv') # 一旦無視

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