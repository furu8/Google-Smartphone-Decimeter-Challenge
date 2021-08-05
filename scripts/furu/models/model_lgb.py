import os
import lightgbm as lgb
# import optuna.integration.lightgbm as lgb
from .model import Model
from .util import Util

# LightGBM
class ModelLGB(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # データのセット
        isvalid = va_x is not None

        # ハイパーパラメータの設定
        params = dict(self.params)
        # num_round = params.pop('num_round')

        model = lgb.LGBMRegressor(**params)

        # 学習
        if isvalid:
            self.model = model.fit(tr_x, tr_y,
                                eval_names=['train', 'valid'],
                                eval_set=[(tr_x, tr_y), (va_x, va_y)],
                                verbose=0,
                                eval_metric=params['metric'],
                                early_stopping_rounds=params['early_stopping_rounds']
                        )
        else:
            self.model = model.fit(tr_x, tr_y)


    def predict(self, te_x):
        return self.model.predict(te_x, num_iteration=self.model.best_iteration_)


    def save_model(self):
        model_path = os.path.join('../../models/lgb', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self):
        model_path = os.path.join('../../models/lgb', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)