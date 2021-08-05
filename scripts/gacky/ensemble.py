# %%
import pandas as pd
import numpy as np
import glob
# %%
sub_df_lat = pd.DataFrame()
sub_df_lng = pd.DataFrame()
sub_df = pd.read_csv('../../data/submission/sample_submission.csv')
#sample_df = pd.read_csv('../../data/processed/ensemble/imu_many_lat_lng_deg_kalman_mean_predict_phone_mean_moving_or_not_PAOnothing.csv')
ensemble_path = "../../data/processed/ensemble/*"
ensemble_df = glob.glob(ensemble_path)
print(ensemble_df)
for path in ensemble_df:
    sub_df_lat = pd.concat([sub_df_lat, pd.read_csv(path).loc[:, ['latDeg']]], axis=1)
    sub_df_lng = pd.concat([sub_df_lng, pd.read_csv(path).loc[:, ['lngDeg']]], axis=1)
# %%
len(ensemble_df)
# %%
sub_df["latDeg"] = sub_df_lat.mean(axis='columns')
sub_df["lngDeg"] = sub_df_lng.mean(axis='columns')
display(sub_df)
# %%
geometric_mean = lambda x: np.abs(np.prod(np.array(x[1:]))) ** (1/len(np.array(x[1:])))
sub_df["latDeg"] = [geometric_mean(i) for i in sub_df_lat.itertuples()]
sub_df["lngDeg"] = [geometric_mean(i) if geometric_mean(i) < 0 else geometric_mean(i) * -1 for i in sub_df_lng.itertuples()]
display(sub_df)
# %%
harmonic_mean = lambda x : np.reciprocal(np.array(x[1:]))
sub_df["latDeg"] = [np.reciprocal(np.mean(harmonic_mean(i))) for i in sub_df_lat.itertuples()]
sub_df["lngDeg"] = [np.reciprocal(np.mean(harmonic_mean(i))) for i in sub_df_lng.itertuples()]
display(sub_df)
# %%
n = 1.2
npow_mean = lambda x, n: np.array(x[1:]) ** n
sub_df["latDeg"] = [np.mean(npow_mean(i, n)) ** (1/n) for i in sub_df_lat.itertuples()]
sub_df["lngDeg"] = [np.mean(npow_mean(i, n)) ** (1/n) for i in sub_df_lng.itertuples()]
display(sub_df)
# %%
weights = []
for idx, path in enumerate(ensemble_df, 1):
    submission = path.split('\\')[1]
    weights.append(float(input(f'{submission}の重みを入力してください {idx}/{len(ensemble_df)}')))
print(ensemble_df)
print(weights)
# %%
weighted_mean = lambda x: np.average(x[1:], weights=weights)
sub_df["latDeg"] = [weighted_mean(i) for i in sub_df_lat.itertuples()]
sub_df["lngDeg"] = [weighted_mean(i) for i in sub_df_lng.itertuples()]
display(sub_df)
# %%
#display(sample_df)
# %%
import plotly.express as px
# %%
fig = px.scatter_mapbox(sub_df, #line_points, #kf_kf_smoothed_baseline[kf_kf_smoothed_baseline["phone"] == "2021-04-22-US-SJC-2_SamsungS20Ultra"],

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
sub_df["lngDeg"].plot()
# %%
sub_df.to_csv('./weighted_mean_lower_4_2_and_non_imu.csv', index=False)
# %%
best_df = pd.read_csv('./weighted_mean_lower_4_2_and_non_imu.csv')
# %%
