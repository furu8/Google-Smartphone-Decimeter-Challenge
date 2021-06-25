import pandas as pd
import numpy as np

from models import Runner, ModelLGB

from IPython.core.display import display

def main():
    phone_name = 'Mi8'

    # pathはinterimではなくprocessedが正しいが動作用のため一旦これで
    train_df = pd.read_csv(f'../../data/processed/confirm/train/2021-06-25_14_{phone_name}.csv')
    # test_df = pd.read_csv(f'../../data/processed/confirm/test/2021-06-25_14_{phone_name}.csv') # 一旦無視

    train_x = train_df.drop(['latDeg', 'lngDeg'], axis=1)
    train_y = train_df[['latDeg', 'lngDeg']]
    run_name = 'lgb'
    lgbm = ModelLGB()
    params = { 
        'max_depth' : 10,
        'num_leaves' : 300,
        'learning_rate' : 0.1,
        'n_estimators': 100,
        'objective': 'regression', # 目的 : 回帰  
        'metric': {'rmse'},        # 評価指標 : rsme(平均二乗誤差の平方根) 
        'verbose' : -1
    }

    runner = Runner(train_x, train_y, run_name, lgbm, params)
    runner.run_train_cv()
    runner.run_predict_cv()


if __name__ == '__main__':
    main()