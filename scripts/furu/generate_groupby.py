# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display
import sys

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
class DataJoiner:
    def __init__(self, phone_name, data_dir='train'):
        self.phone_name = phone_name
        self.data_dir = data_dir

        self.names = ['derived', 'Fix', 
                'OrientationDeg', 'Raw', 'Status', 
                'UncalAccel', 'UncalGyro', 'UncalMag', 'gt']
        if data_dir == 'test':
            self.names.remove('gt')

        self.df_info = pd.DataFrame(index=self.names)

    def load_df_dict(self):
        self.df_dict = {}
        for name in self.names:
            if name == 'derived' or name == 'gt':
                self.df_dict[name] = pd.read_csv(f'../../data/interim/{self.data_dir}/merged_{self.phone_name}_{name}.csv')
            else:
                self.df_dict[name] = pd.read_csv(f'../../data/interim/{self.data_dir}/merged_{self.phone_name}_{name}_add_columns.csv')
    
        return self.df_dict

    def encode_label(self, df, col):
        lenc = LabelEncoder()
        df[col] = lenc.fit_transform(df[col])
        return df

    def encode_CodeType(self):
        self.df_dict['Raw']['CodeType'] = self.df_dict['Raw']['CodeType'].fillna(0)
        for i, category in enumerate(self.df_dict['Raw']['CodeType'].unique()[1:]):
            self.df_dict['Raw'].loc[self.df_dict['Raw']['CodeType']==category, 'CodeType'] = i + 1
    
        self.df_dict['Raw']['CodeType'] = self.df_dict['Raw']['CodeType'].astype(int)

    def groupbys(self, gp_obj, calc_type):
        if calc_type == 'mean':
            return self.groupby_mean(gp_obj)
        elif calc_type == 'max':
            return self.groupby_max(gp_obj)
        elif calc_type == 'min':
            return self.groupby_min(gp_obj)
        elif calc_type == 'median':
            return self.groupby_median(gp_obj)
        elif calc_type == 'mode':
            return self.groupby_mode(gp_obj)
        elif calc_type == 'std':
            return self.groupby_std(gp_obj)
        elif calc_type == 'var':
            return self.groupby_var(gp_obj)
        elif calc_type == 'skew':
            return self.groupby_skew(gp_obj)
        elif calc_type == 'kurt':
            return self.groupby_kurt(gp_obj)
        else:
            raise('calc_typeが規定外')

    def groupby_mean(self, gp_obj):
        return gp_obj.mean()

    def groupby_max(self, gp_obj):
        return gp_obj.max()

    def groupby_min(self, gp_obj):
        return gp_obj.min()

    def groupby_median(self, gp_obj):
        return gp_obj.median()

    def groupby_std(self, gp_obj):
        return gp_obj.std()

    def groupby_var(self, gp_obj):
        return gp_obj.var()

    def groupby_skew(self, gp_obj):
        return gp_obj.skew()
        
    # 使わない
    def groupby_kurt(self, gp_obj):
        return gp_obj.apply(lambda x: x.kurt())

    # 使わない
    def groupby_mode(self, gp_obj):
        return gp_obj.apply(lambda x: x.mode())

    def merge_df(self, df1, df2):
        return pd.merge(df1, df2, on='millisSinceGpsEpoch')

    def merge_df_dict(self, merge_base_col='Raw'):
        df_dict = self.df_dict.copy()
        
        base_df = df_dict[merge_base_col]
        print(base_df.info())
        df_dict.pop(merge_base_col)
        base_df['elapsedRealtimeNanos'] = 0 # suffixesのために、適当な値を入れる
        for key in df_dict.keys():
            if not df_dict[key].empty:
                base_df = pd.merge_asof(base_df, df_dict[key],
                                on='millisSinceGpsEpoch',
                                by=['phoneName', 'collectionName'],
                                suffixes=('', key),
                                direction='nearest',
                                tolerance=1000) # 1sec

        base_df = base_df.drop('elapsedRealtimeNanos', axis=1)

        return base_df

    def drop_all_nan_col(self, key):
        self.df_dict[key] = self.df_dict[key].dropna(axis=1, how='all')

    def interpolate_mean(self, key, col):
        self.df_dict[key][col] = self.df_dict[key][col].fillna(self.df_dict[key][col].mean())

    def sort_df_dict(self):
        order_key = self.df_info[self.df_info['isorder']==False].index
        for key in order_key:
            self.df_dict[key] = self.df_dict[key].sort_values('millisSinceGpsEpoch')
    
    def set_mills_type(self, astype):
        for key in self.df_info[self.df_info['isempty']==False].index:
            self.df_dict[key]['millisSinceGpsEpoch'] = self.df_dict[key]['millisSinceGpsEpoch'].astype(astype)

    def set_df_dict(self, key, df):
        print(f'set: {key}')
        self.df_dict[key] = df.copy()

    def check_millis_order(self):
        diff_millis_list = []
        for key in self.df_dict.keys():
            try:
                diff_millis_list.append(self.df_dict[key][self.df_dict[key]['millisSinceGpsEpoch'].diff()<0].empty)
            except:
                diff_millis_list.append(True) # OrinetationDegにmillisSinceGpsEpochがないとき
        
        self.df_info['isorder'] = diff_millis_list # Trueだと時間順
        print(self.df_info) 

    def check_empty_df(self):
        empty_list = [self.df_dict[key].empty for key in self.df_dict.keys()]
        self.df_info['isempty'] = empty_list
        print(self.df_info)

############################################
        
# %%
def derived(joiner, org_df_dict):
    key = 'derived'

    # Labelig signalType
    labeled_df = joiner.encode_label(org_df_dict[key], 'signalType')

    # gropubys
    groupbyed_df = groupbys(joiner, labeled_df)

    # merge (collectionName, phoneName)
    new_df = merge(joiner, groupbyed_df, org_df_dict, key) # new

    # set
    joiner.set_df_dict(key, new_df)


def fix(joiner, org_df_dict):
    """fixは全部空
    """
    key = 'Fix'
    isempty = org_df_dict[key].empty
    
    if not isempty:
        print(key)
        sys.exit()

    print(f'\n\n\n{key}: {isempty}\n\n\n') # Falseなことあんの？


def orientation_deg(joiner, org_df_dict):
    key = 'OrientationDeg'

    isempty = org_df_dict[key].empty
    
    if not isempty:
        # set
        joiner.set_df_dict(key, org_df_dict[key])
    
    # else:    
    #     print(key)
    #     sys.exit()


def raw(joiner, org_df_dict):
    key = 'Raw'

    # 全欠損カラムを削除
    joiner.drop_all_nan_col(key)
    # 一部欠損を補完
    joiner.interpolate_mean(key, 'AgcDb') 

    # Labeling CodeType
    try:
        joiner.encode_CodeType()
    except:
        print('Pixel4XLModdedはCodeTypeがない')
        pass

     # gropubys
    groupbyed_df = groupbys(joiner, joiner.df_dict[key])

    display(groupbyed_df)

    # merge (collectionName, phoneName)
    new_df = merge(joiner, groupbyed_df, org_df_dict, key) # new

    # set
    joiner.set_df_dict(key, new_df)


def status(joiner, org_df_dict):
    key = 'Status'

    # 全欠損カラムを削除
    joiner.drop_all_nan_col(key)
    # 一部欠損を補完
    joiner.interpolate_mean(key, 'CarrierFrequencyHz')

     # gropubys
    groupbyed_df = groupbys(joiner, joiner.df_dict[key])

    # merge (collectionName, phoneName)
    new_df = merge(joiner, groupbyed_df, org_df_dict, key) # new

    # set
    joiner.set_df_dict(key, new_df)


def uncal_accel(joiner, org_df_dict):
    """OrientationDegとほぼ同じになってしまうが許せ
    """
    key = 'UncalAccel'

    # set
    joiner.set_df_dict(key, org_df_dict[key])


def uncal_gyro(joiner, org_df_dict):
    """OrientationDegとほぼ同じになってしまうが許せ
    """
    key = 'UncalGyro'

    # set
    joiner.set_df_dict(key, org_df_dict[key])


def uncal_mag(joiner, org_df_dict):
    """OrientationDegとほぼ同じになってしまうが許せ
    """
    key = 'UncalMag'

    # set
    joiner.set_df_dict(key, org_df_dict[key])


def ground_truth(joiner):
    key = 'gt'

    # 必要なカラムだけ指定
    columns = ['latDeg', 'lngDeg', 'millisSinceGpsEpoch', 'phoneName', 'collectionName']
    new_df = joiner.df_dict['gt'][columns].copy() # new

    # set
    joiner.set_df_dict(key, new_df)


def merge(joiner, df1, org_df_dict, key):
    df2 = org_df_dict[key].groupby('millisSinceGpsEpoch', as_index=False).first()[['collectionName', 'phoneName', 'millisSinceGpsEpoch']]
    return joiner.merge_df(df1, df2)

def groupbys(joiner, df):
    """今後groupbyを複数回する可能性があるため用意
    """
    calc_type_list = [
        'mean', 'max', 'min', 'median', 
        # 'mode',
        'std', 'var', 
        'skew', 
        # 'kurt'
    ]
    df = df.drop(['phoneName', 'collectionName'], axis=1) # 計算上邪魔なカラムを一旦削る

    # groupybys
    gp_obj = df.groupby('millisSinceGpsEpoch', as_index=False)
    groupbyed_list = [joiner.groupbys(gp_obj, calc_type) for calc_type in calc_type_list]
    
    # display(groupbyed_list[-1])

    # merge
    groupbyed_df = _merge_4groupbys(groupbyed_list, calc_type_list)

    # display(df.info())
    # try:
    #     groupbyed_df = df.groupby('millisSinceGpsEpoch', as_index=False).max()
    # except:
    #     display(type(df['CodeType'][0]))
    #     display(type(df[df['CodeType']==1].iloc[0,0]))
    #     df['CodeType'] = df['CodeType'].astype(int)
    #     groupbyed_df = df.groupby('millisSinceGpsEpoch', as_index=False).max()

    return groupbyed_df

def _merge_4groupbys(groupbyed_list, calc_type_list):
    df = groupbyed_list[0].copy() # mean
    print('mean')
    
    for gped_df, ctype in zip(groupbyed_list[1:], calc_type_list[1:]):
        print(ctype)
        df = pd.merge(df, gped_df, 
                    on='millisSinceGpsEpoch', 
                    suffixes=('', ctype))
    return df
    
##################################### 

def set_df_dict(joiner, org_df_dict, ddir):
    # for key, df in org_df_dict.items():
    #     if not df.empty:
    #         org_df_dict[key]['millisSinceGpsEpoch'] = org_df_dict[key]['millisSinceGpsEpoch'].astype('str')

    # derived
    derived(joiner, org_df_dict)
    # Fix
    # fix(joiner, org_df_dict) # 全部空
    # OrientaionDeg
    orientation_deg(joiner, org_df_dict)
    # Raw
    raw(joiner, org_df_dict)
    # Status
    status(joiner, org_df_dict)
    # UncalAccel
    uncal_accel(joiner, org_df_dict)
    # UncalGyro
    uncal_gyro(joiner, org_df_dict)
    # UncalMag
    uncal_mag(joiner, org_df_dict)
    # ground_truth
    if ddir == 'train':
        ground_truth(joiner)

def merge_df_dict(joiner):
    # 結合カラム紹介
    _show_merged_columns(joiner)

    # millisSinceGpsEpochの型をint64にそろえる
    joiner.set_mills_type(np.int64)
    # millisSinceGpsEpochを時間順にする
    joiner.sort_df_dict()

    return joiner.merge_df_dict()

def _show_merged_columns(joiner):
    print('結合しないカラム')
    for key in joiner.df_info[joiner.df_info['isempty']==True].index:
        print(key)

    print('結合するカラム')
    for key in joiner.df_info[joiner.df_info['isempty']==False].index:
        print(key)

############################################ 

# %%
%%time

phone_names = (
    'Mi8',
    'Pixel4',
    'Pixel4Modded',
    'Pixel4XL',
    'Pixel4XLModded',
    'Pixel5',
    'SamsungS20Ultra'
)

data_dir =  (
    'train',
    'test'
)

for ddir in data_dir:
    for pname in phone_names:
        print(f'\n{ddir}, {pname}\n')

        joiner = DataJoiner(pname, data_dir=ddir)
        # load
        org_df_dict = joiner.load_df_dict() 

        # df empty
        joiner.check_empty_df()
        # millisが正しい順か
        joiner.check_millis_order()
        
        # set
        set_df_dict(joiner, org_df_dict, ddir)

        # merge
        merged_df = merge_df_dict(joiner)

        # save
        merged_df.to_csv(f'../../data/interim/{ddir}/groupbyed_{pname}.csv', index=False)

# # %%
# import pandas as pd
# import numpy as np

# df = pd.DataFrame({'A':['a','b','c','a','b','c','a','b','c','a','b','c'], 
#                     'B':[1,2,3,4,5,6,7,8,9,1,2,3],
#                     'C':[1,np.int64(2),3,4,5,6,7,np.nan,9,4,5,6],
#                     'CodeType':[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,'C','Q','D','P','Q']
# })

# print(df.info())
# print(df)

# df['CodeType'] = df['CodeType'].fillna(0)
# for i, c in enumerate(df['CodeType'].unique()[1:]):
#     df.loc[df['CodeType']==c, 'CodeType'] = i + 1

# print(df.info())
# print(df)

# print(type(df.iloc[0,-1]))
# print(type(df.iloc[-2,-1]))
# print(type(df.iloc[-1,-1]))

# print(df.groupby('A', as_index=False).max())

# df['CodeType'] = df[['CodeType']].astype(int)
# print(df.info())

# # %%
# %%time
# # groupby mode, kurt

# df.groupby('A', as_index=False).apply(pd.DataFrame.kurt)
# # %%
# %%time
# # groupby mode, kurt

# df.groupby('A').apply(lambda x: x.kurt()).reset_index(drop=False)
# # %%
# df.groupby('A', as_index=False).apply(lambda x: x.mode())
# # %%
# df.groupby('A', as_index=False).apply(lambda x: x['B'].value_counts().idxmax())
