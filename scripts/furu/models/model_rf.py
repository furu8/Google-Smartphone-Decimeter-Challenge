import os
from sklearn.ensemble import RandomForestRegressor
from .model import Model
from .util import Util

# LightGBM
class ModelRF(Model):

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        # ハイパーパラメータの設定
        params = dict(self.params)
        # num_round = params.pop('num_round')

        model = RandomForestRegressor(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            random_state=params['random_state'],
            n_jobs=-1
        )

        # 学習
        self.model = model.fit(tr_x, tr_y)

    def predict(self, te_x):
        return self.model.predict(te_x)


    def save_model(self):
        model_path = os.path.join('../../models/rf', f'{self.run_fold_name}.model')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        Util.dump(self.model, model_path)


    def load_model(self):
        model_path = os.path.join('../../models/rf', f'{self.run_fold_name}.model')
        self.model = Util.load(model_path)