# %%
from datetime import time
import pandas as pd
import numpy as np
# %%
from shapely.geometry import Point
import osmnx as ox
import momepy
import geopandas as gpd
# %%
kf_smoothed_baseline = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman.csv')
# %%
gt_test_set = {'../../data/raw/train/2020-09-04-US-SF-2/Mi8/ground_truth.csv': ['2020-05-15-US-MTV-1'], '../../data/raw/train/2020-05-21-US-MTV-2/Pixel4/ground_truth.csv': ['2020-05-28-US-MTV-1'], '../../data/raw/train/2020-05-14-US-MTV-2/Pixel4/ground_truth.csv': ['2020-05-28-US-MTV-2', '2020-06-04-US-MTV-2', '2020-06-10-US-MTV-2'], '../../data/raw/train/2020-06-04-US-MTV-1/Pixel4/ground_truth.csv': ['2020-06-10-US-MTV-1'], '../../data/raw/train/2020-07-08-US-MTV-1/Pixel4/ground_truth.csv': ['2020-08-03-US-MTV-2'], '../../data/raw/train/2020-08-06-US-MTV-2/Mi8/ground_truth.csv': ['2020-08-13-US-MTV-1'], '../../data/raw/train/2021-01-04-US-RWC-1/Pixel4/ground_truth.csv': ['2021-03-16-US-MTV-2'], '../../data/raw/train/2021-04-29-US-TMTV-1/Pixel4/ground_truth.csv': ['2021-03-16-US-RWC-2', '2021-04-21-US-MTV-1', '2021-04-28-US-MTV-2', '2021-04-29-US-MTV-2'], '../../data/raw/train/2021-04-22-US-SJC-1/Pixel4/ground_truth.csv': ['2021-04-22-US-SJC-2', '2021-04-29-US-SJC-3'], '../../data/raw/train/2021-03-10-US-SVL-1/Pixel4XL/ground_truth.csv': ['2021-04-26-US-SVL-2'], '': ['2021-03-25-US-PAO-1', '2021-04-02-US-SJC-1', '2021-04-08-US-MTV-1']}
# %%
gt_test_set = {'../../data/raw/train/2021-04-29-US-TMTV-1/Pixel4/ground_truth.csv': ['2021-04-28-US-MTV-2', '2021-04-29-US-MTV-2'], '../../data/raw/train/2021-04-22-US-SJC-1/Pixel4/ground_truth.csv': ['2021-04-22-US-SJC-2', '2021-04-29-US-SJC-3'], '../../data/raw/train/2021-03-10-US-SVL-1/Pixel4XL/ground_truth.csv': ['2021-04-26-US-SVL-2']}#, '': ['2021-03-25-US-PAO-1', '2021-04-02-US-SJC-1', '2021-04-08-US-MTV-1']}
# %%
def calc_haversine_sub(row, lat2, lon2):
    RADIUS = 6367000
    lat1, lon1, lat2, lon2 = map(np.radians, [row["latDeg"], row["lngDeg"], lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist = 2 * RADIUS * np.arcsin(a ** 0.5)
    return dist
# %%
calc_dist = lambda a, b: grid_points.apply(calc_haversine_sub, lat2=a, lon2=b, axis=1)
# %%
kf_smoothed_baseline = pd.concat([kf_smoothed_baseline, kf_smoothed_baseline['phone'].str.split('_', expand=True).rename(columns={0: 'collectionName', 1: 'phoneName'})], axis=1)
kf_smoothed_baseline
# %%
%%time
#for i in ["2021-04-22-US-SJC-2", "2021-04-29-US-SJC-3"]:#df_collections:#df_collections:
for gt_path, test_list in gt_test_set.items():
    for i in test_list:
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
        grid_points = line_points.loc[:, ['lngDeg', 'latDeg']]
        try:
            gt = pd.read_csv(gt_path)
        except:
            gt = pd.DataFrame({"lngDeg":[], "latDeg":[]})
        display(len(grid_points))
        grid_points = pd.concat([grid_points, gt.loc[:, ['lngDeg', 'latDeg']]], ignore_index=True)
        dist_df = pd.DataFrame(list(map(lambda x: calc_dist(x[3], x[4]), target_df.itertuples())))
        line_points_calc = pd.concat([grid_points, dist_df.T], axis=1)
        line_points_calc = line_points_calc.set_index(["lngDeg", "latDeg"])
        target_df["calc_point"] = line_points_calc.iloc[:,:].idxmin()
        target_df["min_dist"] = line_points_calc.iloc[:, :].min()
        try:
            tmp_df = pd.concat([tmp_df, target_df], ignore_index=True)
        except:
            tmp_df = target_df.copy(deep=True)
#     display(len(line_points))copy(deep=True)
        display(f'{i} done')
# %%
sub_df = pd.DataFrame(list(map(lambda x:float(x[1]), tmp_df["calc_point"])))
sub_df.describe()
# %%
tmp_df["calc_latDeg"] = sub_df
# %%
sub_df = pd.DataFrame(list(map(lambda x:float(x[0]), tmp_df["calc_point"])))
sub_df.describe()
# %%
tmp_df["calc_lngDeg"] = sub_df
# %%
tmp_df["min_dist"].hist(bins=50)#.sort_values(ascending=False).head(2500)
# %%
tmp_df.iloc[1880:1901, :]
# %%
tmp_df.to_csv('../../data/interim/imu_many_lat_lng_deg_kalman_grid_points.csv', index=False)
# %%
tmp_df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman_grid_points.csv')
tmp_df
# %%
sub_df = tmp_df.iloc[:, :7]
# %%
sub_df["latDeg"] = pd.DataFrame(list(map(lambda x: x[10] if x[9] < 20 else x[3], tmp_df.itertuples())))
sub_df["lngDeg"] = pd.DataFrame(list(map(lambda x: x[11] if x[9] < 20 else x[4], tmp_df.itertuples())))
# %%
import plotly.express as px
# %%
sample_df = pd.read_csv('../../data/raw/train/2020-05-21-US-MTV-2/Pixel4/ground_truth.csv')
# %%
fig = px.scatter_mapbox(sub_df[sub_df['phone'] == '2020-05-15-US-MTV-1_Pixel4'], #kf_smoothed_baseline[kf_smoothed_baseline['phone'] == '2020-05-28-US-MTV-1_Pixel4'], #line_points, #kf_kf_smoothed_baseline[kf_kf_smoothed_baseline["phone"] == "2021-04-22-US-SJC-2_SamsungS20Ultra"],

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
# %%
submit_df = pd.read_csv('../../data/interim/imu_many_lat_lng_deg_kalman.csv')
# %%
submit_df = kf_smoothed_baseline
submit_df
# %%
sub_df
# %%
phone_list = sub_df['phone'].unique()
phone_list
# %%
for phone in phone_list:
    submit_df.iloc[submit_df[submit_df['phone'] == phone].index[:],2] = sub_df[(sub_df["collectionName"] == phone.split('_')[0]) & (sub_df["phoneName"] == phone.split('_')[1])]["latDeg"].values
    submit_df.iloc[submit_df[submit_df['phone'] == phone].index[:],3] = sub_df[(sub_df["collectionName"] == phone.split('_')[0]) & (sub_df["phoneName"] == phone.split('_')[1])]["lngDeg"].values
# %%
submit_df.to_csv('../../data/interim/imu_many_lat_lng_deg_kalman_s2gt.csv', index=False)
# %%
