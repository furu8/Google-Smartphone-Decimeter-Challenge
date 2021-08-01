# %%
import plotly.express as px
import pandas as pd
import numpy as np

from IPython.core.display import display

# 最大表示行数を設定
pd.set_option('display.max_rows', 500)
# 最大表示列数の指定
pd.set_option('display.max_columns', 500)

# %%
sample_df = pd.read_csv('../../data/submission/sample_submission.csv')
sample_df = pd.concat([sample_df, sample_df['phone'].str.split('_', expand=True).rename(columns={0:'collectionName', 1:'phoneName'})], axis=1)
sample_df
# %%
name = 'kalman_s2g'
base_test_df = pd.read_csv(f'../../data/interim/{name}.csv')
base_test_df

# %%
# Trueなら下のセル実行OK
(sample_df[['collectionName', 'phoneName']].drop_duplicates() == base_test_df[['collectionName', 'phoneName']].drop_duplicates()).all().all()

# %%
sample_df['latDeg'] = base_test_df['latDeg'].copy()
sample_df['lngDeg'] = base_test_df['lngDeg'].copy()
sample_df
# %%
test_df = pd.read_csv(f'../../data/interim/test/moving_or_not_aga_pred_phone_PAOnothing.csv')
display(test_df)

# %%
# 統計値
cn = '2021-03-16-US-MTV-2'
test_df0 = test_df[(test_df['tag_pred']==0) & (test_df['collectionName']==cn)]
test_df1 = test_df[(test_df['tag_pred']==1) & (test_df['collectionName']==cn)]

print('0: 動')
display(test_df0.describe())
print('1: 停')
display(test_df1.describe())

# %%
test_df = pd.merge_asof(
            test_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg_bs', 'lngDeg_bs', 'tag_pred']].sort_values('millisSinceGpsEpoch'),
            base_test_df[['collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']].sort_values('millisSinceGpsEpoch'),
            on='millisSinceGpsEpoch',
            by=['collectionName', 'phoneName'],
            direction='nearest'
)
test_df

# %%
def scatter_latlng(df):
    fig = px.scatter_mapbox(df,
                        # Here, plotly gets, (x,y) coordinates
                        lat="latDeg",
                        lon="lngDeg",
                        text='phoneName',

                        #Here, plotly detects color of series
                        color="tag_pred",
                        labels="collectionName",

                        zoom=14.5,
                        center={"lat":37.334, "lon":-121.89},
                        height=600,
                        width=800)
    fig.update_layout(mapbox_style='stamen-terrain')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_layout(title_text="GPS trafic")
    fig.show()

# %%
# org
scatter_latlng(test_df)

# %%
cn2pn_df = test_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df
# %%
%%time
new_test_df = test_df.copy()

for cn, pn in cn2pn_df.values:
    idx = 0
    onedf = test_df[(test_df['collectionName']==cn) & (test_df['phoneName']==pn)].copy()
    idxes = onedf.index
    new_lat, new_lng = None, None
    for lat, lng, tag in onedf[['latDeg', 'lngDeg', 'tag_pred']].values:
        if tag < 1:
            new_lat = lat
            new_lng = lng
        elif tag == 1:
            if new_lat is None and new_lng is None:
                pass
            else:
                new_test_df.loc[idxes[idx], 'latDeg'] = new_lat
                new_test_df.loc[idxes[idx], 'lngDeg'] = new_lng
        idx += 1

# %%
display(test_df)
display(new_test_df)
display(new_test_df[new_test_df['collectionName']=='2021-03-16-US-RWC-2'])
# %%
# mon後
scatter_latlng(new_test_df)

# %%
sample_df[sample_df['collectionName']=='2021-03-16-US-RWC-2']

# %%
sample_df[(sample_df['collectionName']=='2021-03-16-US-RWC-2') & (sample_df['phoneName']=='Pixel5')].index
# %%
for cn, pn in cn2pn_df.values:
    sample_df.iloc[sample_df[(sample_df['collectionName']==cn) & (sample_df['phoneName']==pn)].index, 2] = new_test_df.loc[(new_test_df['collectionName']==cn) & (new_test_df['phoneName']==pn), 'latDeg'].values
    sample_df.iloc[sample_df[(sample_df['collectionName']==cn) & (sample_df['phoneName']==pn)].index, 3] = new_test_df.loc[(new_test_df['collectionName']==cn) & (new_test_df['phoneName']==pn), 'lngDeg'].values

# %%
sample_df[sample_df['collectionName']=='2021-03-16-US-RWC-2']
# %%
fig = px.scatter_mapbox(sample_df,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg",
                    lon="lngDeg",
                    # text='phoneName',

                    #Here, plotly detects color of series
                    # color="tag_pred",
                    # labels="collectionName",

                    zoom=14.5,
                    center={"lat":37.334, "lon":-121.89},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()

# %%
# save
sub = pd.read_csv('../../data/submission/sample_submission.csv')
sub = sub.assign(latDeg=sample_df['latDeg'], lngDeg=sample_df['lngDeg'])
sub
# %%
sub.to_csv(f'../../data/interim/{name}_moving_or_not_PAOnothing.csv', index=False)

# %%
fig = px.scatter_mapbox(sub,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg",
                    lon="lngDeg",
                    # text='phoneName',

                    #Here, plotly detects color of series
                    # color="tag_pred",
                    # labels="collectionName",

                    zoom=14.5,
                    center={"lat":37.334, "lon":-121.89},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()
# %%
