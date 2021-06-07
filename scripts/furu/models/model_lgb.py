import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from .model import Model
from .util import Util

# LightGBM
class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        isvalid = va_x is not None
        lgb_train = lgb.Dataset(tr_x, tr_y)
        if isvalid:
            lgb_valid = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        params = dict(self.params)
        # num_round = params.pop('num_round')

        # 学習
        if isvalid:
            self.model = lgb.train(params, lgb_train, valid_sets=lgb_valid)
        else:
            self.model = lgb.train(params, lgb_train)

        # if isvalid:
        #     early_stopping_rounds = params.pop('early_stopping_rounds')
        #     watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        #     self.model = xgb.train(params, dtrain, num_round, evals=watchlist,
        #                            early_stopping_rounds=early_stopping_rounds)
        # else:
        #     watchlist = [(dtrain, 'train')]
        #     self.model = xgb.train(params, dtrain, num_round, evals=watchlist)


    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration)


    def save_model(self):
        model_path = os.path.join('../models/lgb', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self):
        model_path = os.path.join('../models/lgb', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)