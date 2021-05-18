import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from models import ModelLGB, Runner, Util
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb


# LightGBM
def run_lgb(tr_x, tr_y, te_x, te_y, run_fold_name, params, load_path=None):
    lgbm = ModelLGB(run_fold_name, params)
    # 学習
    if load_path is None:
        build_lgb(tr_x, tr_y, lgbm)
    else:
        lgbm.load_model()
    
    # 予測
    pred = predict_lgb(te_x, lgbm, params['objective'])

    # 重要度可視化
    plot_lgb_importance(lgbm)

    # 評価
    print(classification_report(te_y, pred, target_names=['0', '1']))
    print(confusion_matrix(te_y, pred))
    acc = accuracy_score(te_y, pred)
    print(acc)

    return acc


def build_lgb(tr_x, tr_y, lgbm, issave=False):
    lgbm.train(tr_x, tr_y)
    if issave:
        lgbm.save_model()
        print('saved model')


def predict_lgb(te_x, lgbm, objective):
    pred = lgbm.predict(te_x)
    # plt.hist(pred, bins=50)
    # plt.show()
    # pred_median = np.median(pred)
    # pred_median = np.mean(pred)
    # print(pred_median)
    if objective == 'multiclass':
        pred = np.argmax(pred, axis=1)
    elif objective == 'binary':
        # pred = [1 if p > pred_median else 0 for p in pred]
        pred = [1 if p > 0.5 else 0 for p in pred]

    return pred


# 特徴量の重要度を確認
def plot_lgb_importance(lgbm):
    lgb.plot_importance(lgbm.model, height = 0.5, figsize = (4,8))
    plt.show()


def run_cv(train_x, train_y, run_fold_name, params):
    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=2021)
    for tr_idx, va_idx in skf.split(train_x, train_y):
        tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
        va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
        score = run_lgb(tr_x, tr_y, va_x, va_y, run_fold_name, params, load_path=None)
        scores.append(score)

    return np.array(scores)


def main():
    features = Util.load('../config/features/all.pkl')
    
    # データ取得
    df = pd.read_csv('../data/processed/all.csv')
    train_df = df[df['data']=='train'].reset_index(drop=True)
    test_df = df[df['data']=='test'].reset_index(drop=True)

    train_x = train_df[features]
    train_y = train_df['Survived']
    test_x = test_df[features]
    test_y = test_df['Survived']

    # LightGBM
    run_fold_name = 'lgb_all'
    # params = {
    #     'max_depth' : 50,
    #     'num_leaves' : 300,
    #     'learning_rate' : 0.1,
    #     'n_estimators': 100,
    #     'objective':'binary', 
    #     'metric':'binary_logloss', 
    #     # 'metric': 'multi_logloss',
    #     # 'objective': 'multiclass',
    #     # 'num_class' : 5
    #    'verbosity': -1
    # }

    params = {
          'objective' : 'binary', 
          'metric' : 'binary_logloss',
          'verbosity' : -1
    }
    # rub_lgb(train_x, train_y, test_x, test_y, run_fold_name, params)

    scores = run_cv(train_x, train_y, run_fold_name, params)
    print(scores)
    print(scores.mean())

    # runner = Runner(train_x, train_y, 'lgb', ModelLGB, params)
    # runner.run_train_cv()
    # pred = runner.run_predict_cv(test_x)
    # pred_y = [1 if p > 0.5 else 0 for p in pred]
    # # print(pred_y)

    # pred_util = Util.load('../model/pred/lgb-train.pkl')
    # pred_util_y = [1 if p > 0.5 else 0 for p in pred_util]

    # print(len(pred_util_y), len(train_y))

    # # 評価
    # print(classification_report(train_y, pred_util_y, target_names=['0', '1']))

    # # 全体
    # # runner = Runner(train_x, train_y, 'lgb', ModelLGB, params)
    # # runner.run_train_all()
    # # pred = runner.run_predict_all(test_x)
    # # Submission.create_submission('lgb')


if __name__ == "__main__":
    main()