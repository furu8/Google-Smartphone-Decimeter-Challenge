# %%
from datetime import time
import pandas as pd
import numpy as np
# %%
df_test = pd.read_csv("../../data/raw/baseline_locations_test.csv")
df_train = pd.read_csv("../../data/raw/baseline_locations_train.csv")
# %%
df['dist_pre'] = 0
df['dist_pro'] = 0
# %%
def calc_haversine(lat1, lon1, lat2, lon2):
    RADIUS = 6367000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS *np.arcsin(a ** 0.5)
    return dist
# %%
df['latDeg_pre'] = df['latDeg'].shift(periods=1,fill_value=0)
df['lngDeg_pre'] = df['lngDeg'].shift(periods=1,fill_value=0)
df['latDeg_pro'] = df['latDeg'].shift(periods=-1,fill_value=0)
df['lngDeg_pro'] = df['lngDeg'].shift(periods=-1,fill_value=0)
df['dist_pre'] = calc_haversine(df.latDeg_pre, df.lngDeg_pre, df.latDeg, df.lngDeg)
df['dist_pro'] = calc_haversine(df.latDeg, df.lngDeg, df.latDeg_pro, df.lngDeg_pro)

list_phone = df['phone'].unique()
for phone in list_phone:
    ind_s = df[df['phone'] == phone].index[0]
    ind_e = df[df['phone'] == phone].index[-1]
    df.loc[ind_s,'dist_pre'] = 0
    df.loc[ind_e,'dist_pro'] = 0
# %%
pro_95 = df['dist_pro'].mean() + (df['dist_pro'].std() * 2)
pre_95 = df['dist_pre'].mean() + (df['dist_pre'].std() * 2)
ind = df[(df['dist_pro'] > pro_95)&(df['dist_pre'] > pre_95)][['dist_pre','dist_pro']].index

for i in ind:
    df.loc[i,'latDeg'] = (df.loc[i-1,'latDeg'] + df.loc[i+1,'latDeg'])/2
    df.loc[i,'lngDeg'] = (df.loc[i-1,'lngDeg'] + df.loc[i+1,'lngDeg'])/2
# %%
# !pip install simdkalman
# %%
import simdkalman
T = 1.0
state_transition = np.array([[1, 0, T, 0, 0.5 * T ** 2, 0], [0, 1, 0, T, 0, 0.5 * T ** 2], [0, 0, 1, 0, T, 0],
                            [0, 0, 0, 1, 0, T], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
process_noise = np.diag([1e-5, 1e-5, 5e-6, 5e-6, 1e-6, 1e-6]) + np.ones((6, 6)) * 1e-9
observation_model = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
observation_noise = np.diag([5e-5, 5e-5]) + np.ones((2, 2)) * 1e-9

kf = simdkalman.KalmanFilter(
        state_transition = state_transition,
        process_noise = process_noise,
        observation_model = observation_model,
        observation_noise = observation_noise)
# %%
def apply_kf_smoothing(df, kf_=kf):
    unique_paths = df[['collectionName', 'phoneName']].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] == collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]
    return df
# %%
df_sub = pd.read_csv('../../data/submission/sample_submission.csv')
# %%
kf_smoothed_baseline = apply_kf_smoothing(df)
# %%
kf_smoothed_baseline.to_csv('./kf_kf.csv', index=False)
# %%
kf_kf_smoothed_baseline = pd.read_csv('./kf_kf.csv')
# %%
## 無茶苦茶だけどこっからsnap to grid
# !pip install geopandas
# %%
from shapely.geometry import Point
import osmnx as ox
import momepy
import geopandas as gpd
# %%
df_collections = kf_smoothed_baseline['collectionName'].unique()
df_collections
# %%
kf_smoothed_baseline = pd.read_csv('../../data/interim/kalman_aga.csv')
# %%
%%time
for i in ["2021-04-22-US-SJC-2", "2021-04-29-US-SJC-3"]:#df_collections:#df_collections:
    print(i)
    #display(all_df)
    #target_df = df[df['collectionName']==i].reset_index(drop=True)
    target_df = kf_smoothed_baseline[kf_smoothed_baseline['collectionName']==i].reset_index(drop=True)
    #display(target_df)
    target_df["geometry"] = [Point(p) for p in target_df[['lngDeg', 'latDeg']].to_numpy()]
    target_gdf = gpd.GeoDataFrame(target_df, geometry=target_df["geometry"])
    #display(len(target_gdf))
    offset = 0.1**3
    bbox = target_gdf.bounds + [-offset, -offset, offset, offset]
    east = bbox["minx"].min()
    west = bbox["maxx"].max()
    south = bbox["miny"].min()
    north = bbox["maxy"].max()
    G = ox.graph.graph_from_bbox(north, south, east, west, network_type='drive')
    nodes, edges = momepy.nx_to_gdf(G)
    edges = edges.dropna(subset=["geometry"]).reset_index(drop=True)
    hits = target_gdf.bounds.apply(lambda row: list(edges.sindex.intersection(row)), axis=1)
    tmp = pd.DataFrame({
        # index of points table
        "pt_idx": np.repeat(hits.index, hits.apply(len)),
        # ordinal position of line - access via iloc later
        "line_i": np.concatenate(hits.values)
    })
    tmp = tmp.join(edges.reset_index(drop=True), on="line_i")
    tmp = tmp.join(target_gdf.geometry.rename("point"), on="pt_idx")
    tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=target_gdf.crs)
    tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
    tolerance = 0.0005
    tmp = tmp.loc[tmp.snap_dist <= tolerance]
    tmp = tmp.sort_values(by=["snap_dist"])
    closest = tmp.groupby("pt_idx").first()
    closest = gpd.GeoDataFrame(closest, geometry="geometry")
    closest = closest.drop_duplicates("line_i").reset_index(drop=True)
    line_points_list = []
    split = 50  # param: number of split in each LineString
    for dist in range(0, split, 1):
        dist = dist/split
        line_points = closest["geometry"].interpolate(dist, normalized=True)
        line_points_list.append(line_points)
    line_points = pd.concat(line_points_list).reset_index(drop=True)
    line_points = line_points.reset_index().rename(columns={0:"geometry"})
    line_points["lngDeg"] = line_points["geometry"].x
    line_points["latDeg"] = line_points["geometry"].y
    dist_df = pd.DataFrame(list(map(lambda x: calc_dist(x[4], x[5]), target_df.itertuples())))
    line_points_calc = pd.concat([line_points, dist_df.T], axis=1)
    line_points_calc = line_points_calc.set_index(["lngDeg", "latDeg"])
    target_df["calc_point"] = line_points_calc.iloc[:,2:].idxmin()
    target_df["min_dist"] = line_points_calc.iloc[:, 2:].min()
    try:
        tmpo_df = pd.concat([tmpo_df, target_df], ignore_index=True)
    except:
        tmpo_df = target_df.copy(deep=True)
    display(len(line_points))
# %%
df = pd.read_csv('../../data/raw/baseline_locations_test.csv')
# %%
gt = pd.read_csv('../../data/raw/train/2021-04-28-US-TMTV-1/Pixel4/ground_truth.csv')
#gt = pd.concat([gt, pd.read_csv('../../data/raw/train/2021-04-28-US-TMTV-1/Pixel5/ground_truth.csv')], ignore_index=True)
#gt = pd.concat([gt, pd.read_csv('../../data/raw/train/2021-04-28-US-TMTV-1/SamsungS20Ultra/ground_truth.csv')], ignore_index=True)
gt = pd.concat([gt, pd.read_csv('../../data/raw/train/2021-04-29-US-TMTV-1/Pixel4/ground_truth.csv')], ignore_index=True)
#gt = pd.concat([gt, pd.read_csv('../../data/raw/train/2021-04-29-US-TMTV-1/Pixel5/ground_truth.csv')], ignore_index=True)
#gt = pd.concat([gt, pd.read_csv('../../data/raw/train/2021-04-29-US-TMTV-1/SamsungS20Ultra/ground_truth.csv')], ignore_index=True)
# %%
gt
# %%
calc_dist = lambda a, b: gt.apply(calc_haversine_sub, lat2=a, lon2=b, axis=1)
# %%
%%time
for i in ['2021-04-29-US-MTV-2']:#, '2021-04-28-US-MTV-2', '2021-04-29-US-MTV-2']:
    print(i)
    target_df = df[df['collectionName'] == i].reset_index(drop=True)
    dist_df = pd.DataFrame(list(map(lambda x: calc_dist(x[4], x[5]), target_df.itertuples())))
    line_points_calc = pd.concat([gt, dist_df.T], axis=1)
    line_points_calc = line_points_calc.set_index(["lngDeg", "latDeg"])
    target_df["calc_point"] = line_points_calc.iloc[:,2:].idxmin()
    target_df["min_dist"] = line_points_calc.iloc[:, 2:].min()
    try:
        tmp_df = pd.concat([tmp_df, target_df], ignore_index=True)
    except:
        tmp_df = target_df.copy(deep=True)
# %%
closest["geometry"][0]
# %%
tmpo_df
# %%
## こっから下は上でまとめてるやつを1個1個分けたやつ
#kf_smoothed_baseline["geometry"] = [Point(p) for p in kf_smoothed_baseline[["lngDeg", "latDeg"]].to_numpy()]
df["geometry"] = [Point(p) for p in df[["lngDeg", "latDeg"]].to_numpy()]
#target_gt_gdf = gpd.GeoDataFrame(kf_smoothed_baseline, geometry=kf_smoothed_baseline["geometry"])
target_gt_gdf = gpd.GeoDataFrame(df, geometry=df["geometry"])
target_gt_gdf
# %%
target_gt_gdf#.bounds
# %%
offset = 0.1**3
bbox = target_gt_gdf.bounds + [-offset, -offset, offset, offset]
# %%
east = bbox["minx"].min()
west = bbox["maxx"].max()
south = bbox["miny"].min()
north = bbox["maxy"].max()
# %%
G = ox.graph.graph_from_bbox(north, south, east, west, network_type='all_private')
# %%
G
# %%
nodes, edges = momepy.nx_to_gdf(G)
# %%
nodes
# %%
edges
# %%
edges = edges.dropna(subset=["geometry"]).reset_index(drop=True)
# %%
edges.sindex.intersection([-122.081959, 37.416611, -122.081959, 37.416611])
# %%
hits = target_gt_gdf.bounds.apply(lambda row: list(edges.sindex.intersection(row)), axis=1)
# %%
bbox
# %%
np.concatenate([[76555, 86260], [76555, 86260]])#_zero = hits.apply(len).replace(0, 1)

# %%
for i, l in enumerate(hits.values):
    if len(l) == 0:
        print(i, l)
# %%
bbox.iloc[14003, :]
# %%
tmp = pd.DataFrame({
    # index of points table
    "pt_idx": np.repeat(hits.index, hits.apply(len)),
    # ordinal position of line - access via iloc later
    "line_i": np.concatenate(hits.values)
})
# %%
tmp
# %%
# Join back to the lines on line_i; we use reset_index() to
# give us the ordinal position of each line
tmp = tmp.join(edges.reset_index(drop=True), on="line_i")
# %%
len(tmp['pt_idx'].unique())
# %%
# Join back to the original points to get their geometry
# rename the point geometry as "point"
tmp = tmp.join(target_gt_gdf.geometry.rename("point"), on="pt_idx")
tmp
# %%
display(target_gt_gdf)
display(edges)
# %%
# Convert back to a GeoDataFrame, so we can do spatial ops
tmp = gpd.GeoDataFrame(tmp, geometry="geometry", crs=target_gt_gdf.crs)
tmp
# %%
tmp["snap_dist"] = tmp.geometry.distance(gpd.GeoSeries(tmp.point))
tmp
# %%
# Discard any lines that are greater than tolerance from points
tolerance = 0.0005
tmp = tmp.loc[tmp.snap_dist <= tolerance]
tmp
# %%
# Sort on ascending snap distance, so that closest goes to top
tmp = tmp.sort_values(by=["snap_dist"])
tmp
# %%
# group by the index of the points and take the first, which is the
# closest line
closest = tmp.groupby("pt_idx").first()
closest
# %%
# construct a GeoDataFrame of the closest lines
closest = gpd.GeoDataFrame(closest, geometry="geometry")
closest = closest.drop_duplicates("line_i").reset_index(drop=True)
# %%
closest.head(5)
# %%
line_points_list = []
split = 50  # param: number of split in each LineString
for dist in range(0, split, 1):
    dist = dist/split
    line_points = closest["geometry"].interpolate(dist, normalized=True)
    line_points_list.append(line_points)
line_points = pd.concat(line_points_list).reset_index(drop=True)
line_points = line_points.reset_index().rename(columns={0:"geometry"})
line_points["lngDeg"] = line_points["geometry"].x
line_points["latDeg"] = line_points["geometry"].y
# %%
line_points
## 分けたやつはここまで
# %%
df
# %%
display(line_points.latDeg)
display(kf_smoothed_baseline.latDeg)
# %%
submit_df = pd.read_csv('../../data/interim/aga_mean_predict_phone_mean.csv')
# %%
## 提出ファイル作るやつ
df_sub = df_sub.assign(
    latDeg = submit_df.latDeg,
    lngDeg = submit_df.lngDeg
)
df_sub.to_csv('./submission9.csv', index=False)
# %%
import plotly.express as px
# %%
df_train_phone = df_train['phone'].unique()
df_test_phone = df_test['phone'].unique()
# %%
df_train_phone
# %%
df_test_phone
# %%
# for phone in df_train_phone:
fig = px.scatter_mapbox(gt, #df_test[df_test['phone'] == '2020-08-03-US-MTV-2_Pixel4'], #line_points, #df[(df["collectionName"] == "2021-04-26-US-SVL-2")], #line_points, #kf_kf_smoothed_baseline[kf_kf_smoothed_baseline["phone"] == "2021-04-22-US-SJC-2_SamsungS20Ultra"],

                    # Here, plotly gets, (x,y) coordinates
                    lat="latDeg",
                    lon="lngDeg",

                    zoom=15,
                    center={"lat":37.33351, "lon":-121.8906},
                    height=600,
                    width=800)
fig.update_layout(mapbox_style='stamen-terrain')
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.update_layout(title_text="GPS trafic")
fig.show()
#fig.write_image('./hokan.png')
# %%
kf_kf_smoothed_baseline[kf_kf_smoothed_baseline["collectionName"] == "2021-04-22-US-SJC-2"]["phoneName"].unique()
# %%
line_points_calc = line_points.copy(deep=True)
# %%
calc_haversine(line_points_calc["latDeg"], line_points_calc["lngDeg"], df["latDeg"], df["lngDeg"])
# %%
min([calc_haversine(lat, lng, df.at[0, "latDeg"], df.at[0, "lngDeg"]) for lat, lng in zip(line_points_calc["latDeg"], line_points["lngDeg"])])
# %%
min([calc_haversine(lat, lng, df.at[1, "latDeg"], df.at[1, "lngDeg"]) for lat, lng in zip(line_points_calc["latDeg"], line_points["lngDeg"])])
# %%
dist_df = [[calc_haversine(lat, lng, df.at[i, "latDeg"], df.at[i, "lngDeg"]) for lat, lng in zip(line_points_calc["latDeg"], line_points["lngDeg"])] for i in range(len(df))]
# %%
target_df
# %%
calc_dist = lambda a, b: line_points.apply(calc_haversine_sub, lat2=a, lon2=b, axis=1)
# %%
dist_df = pd.DataFrame(list(map(lambda x: calc_dist(x[4], x[5]), target_df.itertuples())))
# %%
line_points_calc = pd.concat([line_points, dist_df.T], axis=1)
# %%
line_points_calc = line_points_calc.set_index(["lngDeg", "latDeg"])
# %%
target_df["calc_point"] = line_points_calc.iloc[:,2:].idxmin()
# %%
target_df["min_dist"] = line_points_calc.iloc[:, 2:].min()
# %%
target_df
all_df = target_df.copy(deep=True)
# %%
all_df = pd.concat([all_df, target_df], ignore_index=True)
# %%
all_df.describe()
# %%
tmpo_df = pd.read_csv('../../data/interim/kalman_mean_predict_snap.csv')
# %%
sub_df = pd.DataFrame(list(map(lambda x:x[1], tmp_df["calc_point"])))
sub_df.describe()
# %%
tmp_df["calc_latDeg"] = sub_df
# %%
sub_df = pd.DataFrame(list(map(lambda x:x[0], tmp_df["calc_point"])))
sub_df.describe()
# %%
tmp_df["calc_lngDeg"] = sub_df
# %%
tmp_df["min_dist"].plot()
# %%
def calc_haversine_sub(row, lat2, lon2):
    RADIUS = 6367000
    lat1, lon1, lat2, lon2 = map(np.radians, [row["latDeg"], row["lngDeg"], lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS *np.arcsin(a ** 0.5)
    return dist
# %%
tmp_df
# %%
sub_df = tmp_df.iloc[:, :3]
# %%
sub_df["latDeg"] = pd.DataFrame(list(map(lambda x: x[10] if x[9] > 10 else x[4], tmp_df.itertuples())))
sub_df["lngDeg"] = pd.DataFrame(list(map(lambda x: x[11] if x[9] > 10 else x[5], tmp_df.itertuples())))
# %%
sub_df = pd.DataFrame(columns=['latDeg', 'lngDeg'])
# %%
sub_df = sub_df.rename({0:"latDeg"}, axis=1)
# %%
sub_df
# %%
all_df.to_csv('./kf_kf_grid_df.csv', index=False)
# %%
all_df = pd.read_csv("./grid_df.csv", index_col=0)
# %%
tmp_df.to_csv('../../data/interim/aga_mean_predict_snap.csv', index=False) #.describe()#["min_dist"].plot()
# %%
def get_removedevice(input_df: pd.DataFrame, divece: str) -> pd.DataFrame:
    input_df['index'] = input_df.index
    input_df = input_df.sort_values('millisSinceGpsEpoch')
    input_df.index = input_df['millisSinceGpsEpoch'].values

    output_df = pd.DataFrame()
    for _, subdf in input_df.groupby('collectionName'):

        phones = subdf['phoneName'].unique()

        if (len(phones) == 1) or (not divece in phones):
            output_df = pd.concat([output_df, subdf])
            continue

        origin_df = subdf.copy()

        _index = subdf['phoneName']==divece
        subdf.loc[_index, 'latDeg'] = np.nan
        subdf.loc[_index, 'lngDeg'] = np.nan
        # subdf = subdf.interpolate(method='index', limit_area='inside')
        subdf.interpolate(method='index', limit_area='inside', inplace=True)

        _index = subdf['latDeg'].isnull()
        subdf.loc[_index, 'latDeg'] = origin_df.loc[_index, 'latDeg'].values
        subdf.loc[_index, 'lngDeg'] = origin_df.loc[_index, 'lngDeg'].values

        output_df = pd.concat([output_df, subdf])

    output_df.index = output_df['index'].values
    output_df = output_df.sort_index()

    del output_df['index']

    return output_df
# %%
train_remove = get_removedevice(kf_kf_smoothed_baseline, 'SamsungS20Ultra')
# %%
train_remove
# %%
sample_df = tmp_df.iloc[:, :3]
# %%
sample_df = pd.concat([sample_df, sub_df], axis=1)
# %%
sample_df.to_csv('./kf_kf_grid_sample.csv', index=False)
# %%
tmpo_df["min_dist"].describe()
# %%
submit_df = pd.read_csv('../../data/interim/kalman_aga.csv')
# %%
submit_df
# %%
tmpo_df
# %%
sub_df_20210422_US_SJC_2 = sub_df.copy(deep=True)
# %%
sub_df_20210422_US_SJC_2.to_csv("grid_20210422_US_SJC_2_Samsung.csv", index=False)
# %%
print(pd.merge(submit_df, sub_df_20210422_US_SJC_2, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='left'))
# %%
print(pd.merge(submit_df, sub_df, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='left'))
# %%
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-29-US-SJC-3_SamsungS20Ultra"].index[:],3] = sub_df[(sub_df["collectionName"] == "2021-04-29-US-SJC-3") & (sub_df["phoneName"] == "SamsungS20Ultra")]["latDeg"].values
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-29-US-SJC-3_SamsungS20Ultra"].index[:],4] = sub_df[(sub_df["collectionName"] == "2021-04-29-US-SJC-3") & (sub_df["phoneName"] == "SamsungS20Ultra")]["lngDeg"].values
# %%
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-29-US-SJC-3_Pixel4"].index[:],3] = sub_df[(sub_df["collectionName"] == "2021-04-29-US-SJC-3") & (sub_df["phoneName"] == "Pixel4")]["latDeg"].values
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-29-US-SJC-3_Pixel4"].index[:],4] = sub_df[(sub_df["collectionName"] == "2021-04-29-US-SJC-3") & (sub_df["phoneName"] == "Pixel4")]["lngDeg"].values
# %%
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-22-US-SJC-2_SamsungS20Ultra"].index[:],3] = sub_df[(sub_df["collectionName"] == "2021-04-22-US-SJC-2") & (sub_df["phoneName"] == "SamsungS20Ultra")]["latDeg"].values
submit_df.iloc[submit_df[submit_df['phone'] == "2021-04-22-US-SJC-2_SamsungS20Ultra"].index[:],4] = sub_df[(sub_df["collectionName"] == "2021-04-22-US-SJC-2") & (sub_df["phoneName"] == "SamsungS20Ultra")]["lngDeg"].values
# %%
submit_df.to_csv('../../data/interim/kalman_aga_s2g_drive.csv', index=False)
# %%
