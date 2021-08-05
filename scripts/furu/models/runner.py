import numpy as np
import pandas as pd
from .model import Model
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from typing import Callable, List, Optional, Tuple, Union

from .util import Logger, Util

# logger = Logger()


class Runner:

    # def __init__(self, df, run_name: str, model_cls: Callable[[str, dict], Model], features: List[str], params: dict):
    def __init__(self, train_x, train_y, run_name: str, 
                latmodel: Callable[[str, dict], Model], lngmodel: Callable[[str, dict], Model], 
                params: dict):
        """コンストラクタ

        :param run_name: ランの名前
        :param latmodel: 緯度モデル
        :param lngmodel: 経度モデル
        :param features: 特徴量のリスト
        :param params: ハイパーパラメータ
        """
        self.train_x = train_x
        self.train_y = train_y

        self.run_name = run_name
        self.latmodel = latmodel
        self.lngmodel = lngmodel
        # self.features = features
        self.params = params
        self.n_fold = 4

    def train_fold(self, i_fold: Union[int, str]) -> Tuple[
        Model, Optional[np.array], Optional[np.array], Optional[float]]:
        """クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号（すべてのときには'all'とする）
        :return: （モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア）のタプル
        """
        # 学習データの読込
        isvalid = i_fold != 'all'
        # train_x = self.load_x_train()
        # train_y = self.load_y_train()
        train_x = self.train_x
        train_y = self.train_y


        score_df = pd.DataFrame()

        if isvalid:
            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self.load_index_fold(i_fold)
            tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
            va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]

            # 学習を行う
            latmodel = self.build_model(i_fold, isDeg='lat')
            lngmodel = self.build_model(i_fold, isDeg='lng')
            latmodel.train(tr_x, tr_y['latDeg'], va_x, va_y['latDeg'])
            lngmodel.train(tr_x, tr_y['lngDeg'], va_x, va_y['lngDeg'])

            # バリデーションデータへの予測・評価を行う
            lat_va_pred = latmodel.predict(va_x)
            lng_va_pred = lngmodel.predict(va_x)

            score_df['lat_pred'] = lat_va_pred
            score_df['lng_pred'] = lng_va_pred
            score_df['lat_truth'] = va_y['latDeg'].values
            score_df['lng_truth'] = va_y['lngDeg'].values
            score_df['dist'] = self.evaluate_lat_lng_dist(score_df)
            
            # モデル、スコアを返す
            return latmodel, lngmodel, score_df
        
        else:
            # 学習データ全てで学習を行う
            latmodel = self.build_model(i_fold, isDeg='lat')
            lngmodel = self.build_model(i_fold, isDeg='lng')
            latmodel.train(train_x, train_y['latDeg'])
            lngmodel.train(train_x, train_y['lngDeg'])

            # モデルを返す
            return latmodel, lngmodel, None

    def run_train_cv(self) -> None:
        """クロスバリデーションでの学習・評価を行う

        学習・評価とともに、各foldのモデルの保存、スコアのログ出力についても行う
        """
        # logger.info(f'{self.run_name} - start training cv')

        score_df_list = []

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            # logger.info(f'{self.run_name} fold {i_fold} - start training')
            latmodel, lngmodel, score_df = self.train_fold(i_fold)
            # logger.info(f'{self.run_name} fold {i_fold} - end training - score {score}')

            # モデルを保存する
            latmodel.save_model()
            lngmodel.save_model()

            # 結果を保持する
            score_df_list.append(score_df)


        # logger.info(f'{self.run_name} - end training cv - score {np.mean(scores)}')

        # 予測結果の保存
        # Util.dump(latpreds, f'../model/pred/{self.run_name}-train.pkl')

        # 評価結果の保存
        # logger.result_scores(self.run_name, scores)

        return score_df_list

    def run_predict_cv(self, test_x) -> None:
        """クロスバリデーションで学習した各foldのモデルの平均により、テストデータの予測を行う

        あらかじめrun_train_cvを実行しておく必要がある
        """
        # logger.info(f'{self.run_name} - start prediction cv')

        # test_x = self.load_x_test()

        latpreds = []
        lngpreds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            # logger.info(f'{self.run_name} - start prediction fold:{i_fold}')
            latmodel = self.build_model(i_fold, isDeg='lat')
            lngmodel = self.build_model(i_fold, isDeg='lng')
            latmodel.load_model()
            lngmodel.load_model()
            latpred = latmodel.predict(test_x)
            lngpred = lngmodel.predict(test_x)

            latpreds.append(latpred)
            lngpreds.append(lngpred)

            # logger.info(f'{self.run_name} - end prediction fold:{i_fold}')

        # 予測の平均値を出力する
        latpred_avg = np.mean(latpreds, axis=0)
        lngpred_avg = np.mean(lngpreds, axis=0)

        # 予測結果の保存
        Util.dump(latpred_avg, f'../../result/{self.run_name}-test.pkl')
        Util.dump(lngpred_avg, f'../../result/{self.run_name}-test.pkl')

        # logger.info(f'{self.run_name} - end prediction cv')

    
    def evaluate_lat_lng_dist(self, df):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # Radius of earth in kilometers is 6371 or 6371
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

    def run_train_all(self) -> None:
        """学習データすべてで学習し、そのモデルを保存する"""
        # logger.info(f'{self.run_name} - start training all')

        # 学習データ全てで学習を行う
        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        # logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self, test_x) -> None:
        """学習データすべてで学習したモデルにより、テストデータの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        # logger.info(f'{self.run_name} - start prediction all')

        # test_x = self.load_x_test()
        # test_x = self.test_x

        # 学習データ全てで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold)
        model = self.build_model(i_fold)
        model.load_model()
        pred = model.predict(test_x)

        # 予測結果の保存
        Util.dump(pred, f'../model/pred/{self.run_name}-test.pkl')

        # logger.info(f'{self.run_name} - end prediction all')

        return pred

    def build_model(self, i_fold: Union[int, str], isDeg) -> Model:
        """クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param i_fold: foldの番号
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        if isDeg == 'lat':
            run_fold_name = f'{self.run_name}-lat-{i_fold}'
            return self.latmodel(run_fold_name, self.params)
        elif isDeg == 'lng':
            run_fold_name = f'{self.run_name}-lng-{i_fold}'
            return self.lngmodel(run_fold_name, self.params)
        else:
            raise Exception('isDegミスってね？')


    def load_x_train(self) -> pd.DataFrame:
        """学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データの読込を行う
        # 列名で抽出する以上のことを行う場合、このメソッドの修正が必要
        # 毎回train.csvを読み込むのは効率が悪いため、データに応じて適宜対応するのが望ましい（他メソッドも同様）
        return pd.read_csv('../input/train.csv')[self.features]

    def load_y_train(self) -> pd.Series:
        """学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 目的変数の読込を行う
        train_y = pd.read_csv('../input/train.csv')['target']
        train_y = np.array([int(st[-1]) for st in train_y]) - 1
        train_y = pd.Series(train_y)
        return train_y

    def load_x_test(self) -> pd.DataFrame:
        """テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        return pd.read_csv('../input/test.csv')[self.features]

    def load_index_fold(self, i_fold: int) -> np.array:
        """クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: foldに対応するレコードのインデックス
        """
        # 学習データ・バリデーションデータを分けるインデックスを返す
        # ここでは乱数を固定して毎回作成しているが、ファイルに保存する方法もある
        # train_y = self.load_y_train()
        train_y = self.train_y
        dummy_x = np.zeros(len(train_y))
        tss = TimeSeriesSplit(n_splits=self.n_fold)
        return list(tss.split(dummy_x, train_y))[i_fold]
