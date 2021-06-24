# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display

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

# %%
# signalType
lenc = LabelEncoder()
train_dr_df['signalType'] = lenc.fit_transform(train_dr_df['signalType'])
# %%
# groupy by for derived
derived_gb_list = ['phoneName', 'collectionName', 'ConstellationType', 'Svid', 'signalType']

train_dr_df_4groupby = train_dr_df.drop(derived_gb_list, axis=1)
train_dr_df_category = train_dr_df[['millisSinceGpsEpoch']+derived_gb_list]

# %%
# gropubyする
train_dr_df_mean = train_dr_df_4groupby.groupby('millisSinceGpsEpoch', as_index=False).mean()
train_dr_df_mean

# %%
# category
constellation_type_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='ConstellationType'), columns=['ConstellationType']).reset_index()
svid_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='Svid'), columns=['Svid']).reset_index()
signal_type_df = pd.DataFrame(train_dr_df_category.groupby(['millisSinceGpsEpoch']).apply(calc_max_index, col='signalType'), columns=['signalType']).reset_index()
# %%
display(constellation_type_df)
display(svid_df)
display(signal_type_df)

# %%
train_dr_df_mean['ConstellationType'] = constellation_type_df['ConstellationType'].values
train_dr_df_mean['Svid'] = svid_df['Svid'].values
train_dr_df_mean['signalType'] = signal_type_df['signalType'].values
train_dr_df_mean

# %%
new_train_dr_df = pd.merge_asof(train_dr_df_mean, train_dr_df[['phoneName', 'collectionName', 'millisSinceGpsEpoch']], on='millisSinceGpsEpoch')
new_train_dr_df
# %%
# lat（緯度）
display(train_dr_df)

# runner = Runner()
