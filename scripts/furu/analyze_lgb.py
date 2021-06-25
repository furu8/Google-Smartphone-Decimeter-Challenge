# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
def calc_max_index(df, col):
    return df[col].value_counts().idxmax()

# %%
phone_name = input('スマホの名前指定: ')

train_dr_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_derived.csv')
train_gnss_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_gnss.csv')
train_gt_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_gt.csv')
test_dr_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_derived.csv')
test_gnss_df = pd.read_csv(f'../../data/interim/train/merged_{phone_name}_gnss.csv')

# shapeだけ表示
display(train_dr_df.shape)
display(train_gnss_df.shape)
display(train_gt_df.shape)
display(test_dr_df.shape)
display(test_gnss_df.shape)

# %%[markdown]
# # derived
# %%
# signalType
lenc = LabelEncoder()
train_dr_df['signalType'] = lenc.fit_transform(train_dr_df['signalType'])
# %%
# groupy by for derived
derived_list = ['phoneName', 'collectionName', 'ConstellationType', 'Svid', 'signalType']

train_dr_df_4groupby = train_dr_df.drop(derived_list, axis=1)
train_dr_df_category = train_dr_df[['millisSinceGpsEpoch']+derived_list]

# %%
# gropubyする
train_dr_df_mean = train_dr_df_4groupby.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_dr_df_mean

# %%
# category変数はもっとも多い値
constellation_type_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='ConstellationType'), columns=['ConstellationType']).reset_index()
svid_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='Svid'), columns=['Svid']).reset_index()
signal_type_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='signalType'), columns=['signalType']).reset_index()
# %%
display(constellation_type_df)
display(svid_df)
display(signal_type_df)

# %%
# category変数をgroupbyした数値と結合
train_dr_df_mean['ConstellationType'] = constellation_type_df['ConstellationType'].values
train_dr_df_mean['Svid'] = svid_df['Svid'].values
train_dr_df_mean['signalType'] = signal_type_df['signalType'].values
train_dr_df_mean

# %%
# phoneNameとcollectionNameを結合
new_train_dr_df = pd.merge_asof(train_dr_df_mean, train_dr_df[['phoneName', 'collectionName', 'millisSinceGpsEpoch']], on='millisSinceGpsEpoch')
new_train_dr_df

# %%[markdown]
# # gnss

# %%
# CodeType
train_gnss_df['CodeType'].unique()
# %%
train_gnss_df = train_gnss_df.fillna(0)
train_gnss_df.loc[train_gnss_df['CodeType']=='C', 'CodeType'] = 1
train_gnss_df.loc[train_gnss_df['CodeType']=='Q', 'CodeType'] = 2
train_gnss_df[['CodeType']]

# %%
train_gnss_df['HasEphemerisData'].unique()

# %%
# groupy by for gnss
gnss_list = ['Svid', 
            'AccumulatedDeltaRangeState',
            'MultipathIndicator',
            'ConstellationType',
            'ConstellationTypeStatus',
            'SvidStatus',
            'AzimuthDegrees',
            'ElevationDegrees',
            'UsedInFix',
            'HasAlmanacData',
            'HasEphemerisData',
            'CodeType'
            ]

train_gnss_df_4groupby = train_gnss_df.drop(gnss_list, axis=1)
train_gnss_df_category = train_gnss_df[['millisSinceGpsEpoch']+gnss_list]

# %%
# gropubyする
train_gnss_df_mean = train_gnss_df_4groupby.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_gnss_df_mean

# %%
# category変数はもっとも多い値
for gnss_col in gnss_list:
    df = pd.DataFrame(train_gnss_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col=gnss_col), columns=[gnss_col]).reset_index()
    train_gnss_df_mean[gnss_col] = df[gnss_col].values
# %%
display(train_gnss_df_mean)

# %%
# phoneNameとcollectionNameを結合
new_train_gnss_df = pd.merge_asof(train_gnss_df_mean, 
                            train_dr_df[['phoneName', 'collectionName', 'millisSinceGpsEpoch']], 
                            on='millisSinceGpsEpoch', 
                            direction='nearest')
new_train_gnss_df

# %%[markdown]
# # ground_truth
# %%
train_gt_df

# %%[markdown]
# # 結合

# %%
# 結合メンバー
display(new_train_dr_df)
display(new_train_gnss_df)
display(train_gt_df)

# %%
train_gnssgt_df = pd.merge_asof(new_train_gnss_df, train_gt_df, 
                        on='millisSinceGpsEpoch',
                        by=['phoneName', 'collectionName'],
                        direction='nearest',
                        tolerance=1000
                        )
train_df = pd.merge_asof(train_gnssgt_df, new_train_dr_df, 
                    on = 'millisSinceGpsEpoch',
                    by=['phoneName', 'collectionName'],
                    suffixes=('_gnss', '_derived'),
                    direction='nearest',
                    tolerance=10000 # Mi8は1000にすると欠損
                    )
train_df

# %%
# tolerance=1000にしたとき、derivedと結合する欠損するやつ
train_df.loc[train_df['Svid_derived'].isnull(), ['millisSinceGpsEpoch']]

# %%
train_df[3835:3840]

# %%
