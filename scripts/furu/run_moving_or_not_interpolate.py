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
sample_df
# %%
test_df = pd.read_csv('../../data/interim/test/moving_or_not.csv')
display(test_df)

# %%
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
%%time
new_test_df = test_df.copy()
idx = 0
for lat, lng, tag in test_df[['latDeg', 'lngDeg', 'tag_pred']].values:
    if tag < 1:
        new_lat = lat
        new_lng = lng
    elif tag == 1:
        new_test_df.loc[idx, 'latDeg'] = new_lat
        new_test_df.loc[idx, 'lngDeg'] = new_lng
    idx += 1

# %%
display(test_df)
display(new_test_df)
# %%
fig = px.scatter_mapbox(new_test_df,
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