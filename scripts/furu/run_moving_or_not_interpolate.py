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
sample_df = pd.read_csv('../../data/interim/aga_mean_predict_phone_mean_test.csv')
sample_df
# %%
test_df = pd.read_csv('../../data/interim/test/moving_or_not_aga_pred_phone.csv')
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
fig = px.scatter_mapbox(test_df,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg_bs",
                    lon="lngDeg_bs",
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
%%time
new_test_df = test_df.copy()
idx = 0
for lat, lng, tag in test_df[['latDeg_bs', 'lngDeg_bs', 'tag_pred']].values:
    if tag < 1:
        new_lat = lat
        new_lng = lng
    elif tag == 1:
        try:
            new_test_df.loc[idx, 'latDeg_bs'] = new_lat
            new_test_df.loc[idx, 'lngDeg_bs'] = new_lng
        except NameError:
            pass
    idx += 1

# %%
display(test_df)
display(new_test_df)
# %%
fig = px.scatter_mapbox(new_test_df,
                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg_bs",
                    lon="lngDeg_bs",
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
cn2pn_df = new_test_df[['collectionName', 'phoneName']].drop_duplicates()
cn2pn_df
# %%
for cn, pn in cn2pn_df.values:
    sample_df.iloc[sample_df[(sample_df['collectionName']==cn) & (sample_df['phoneName']==pn)].index, 3] = new_test_df.loc[(new_test_df['collectionName']==cn) & (new_test_df['phoneName']==pn), 'latDeg_bs'].values
    sample_df.iloc[sample_df[(sample_df['collectionName']==cn) & (sample_df['phoneName']==pn)].index, 4] = new_test_df.loc[(new_test_df['collectionName']==cn) & (new_test_df['phoneName']==pn), 'lngDeg_bs'].values

# %%
sample_df
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
sub.to_csv('../../data/submission/moving_or_not_aga_pred_phone.csv', index=False)

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