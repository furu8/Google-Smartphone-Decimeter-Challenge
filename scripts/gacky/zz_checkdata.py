#%%
import numpy as np
from numpy.lib.function_base import disp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, display_pdf
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns
# %%
## データ読み込み
submission = pd.read_csv('../../data/submission/sample_submission.csv')
baseline_train = pd.read_csv('../../data/raw/baseline_locations_train.csv')
raw_data = pd.read_csv('../../data/raw/train/2020-07-17-US-MTV-1/Mi8/Mi8_derived.csv')
raw_data_test = pd.read_csv('../../data/raw/test/2020-05-28-US-MTV-1/Pixel4XL/Pixel4XL_derived.csv')
gt = pd.read_csv('../../data/raw/train/2020-07-17-US-MTV-1/Mi8/ground_truth.csv')
# %%
display(raw_data["millisSinceGpsEpoch"].unique().shape)
display(gt.shape)
# %%
display(raw_data)
display(gt)
# %%
raw_data['millisSinceGpsEpoch'].plot()
# %%
gt['millisSinceGpsEpoch'].plot()
# %%
display(raw_data.groupby('millisSinceGpsEpoch').mean())
# %%
## merge_asofで結合マン
merge_df = pd.merge_asof(raw_data, gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", allow_exact_matches=True)
display(merge_df)
# %%
display(merge_df.describe())
display(merge_df.info())
# %%
display(raw_data.dtypes)
display(raw_data_test.dtypes)
display(gt.dtypes)
# %%
#!pip install pynmea2
# %%
import pynmea2
# %%
## nmeaをdataframeにしよう
with open("../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/supplemental/SPAN_Pixel4_10Hz.nmea", encoding='utf-8') as nmea_f:
    for line in nmea_f.readlines():
        try:
            msg = pynmea2.parse(line)
            break
        except pynmea2.ParseError as e:
            pass
        
print(repr(msg))
print("---------------------------------")
print("timestamp:", msg.timestamp)
print("lat:", msg.lat)
print("lat_dir:", msg.lat_dir)
print("lon:", msg.lon)
print("lon_dir:", msg.lon_dir)
print("gps_qual:", msg.gps_qual)
print("num_sats:", msg.num_sats)
print("horizontal_dil:", msg.horizontal_dil)
print("altitude:", msg.altitude)
print("altitude_units:", msg.altitude_units)
print("geo_sep:", msg.geo_sep)
print("geo_sep_units:", msg.geo_sep_units)
print("age_gps_data:", msg.age_gps_data)
print("ref_station_id:", msg.ref_station_id)
print("---------------------------------")
# %%
def load_nmea_file(nmea_path):
    """Convert GGA sentences of nmea files to pandas dataframe.
    """
    
    gga_lines = []
    with open(nmea_path, encoding='utf-8') as nmea_f:
        for line in nmea_f.readlines():
            try:
                msg = pynmea2.parse(line)
                gga_lines.append(msg)
            except pynmea2.ParseError as e:
                pass
            
      
    return pd.DataFrame(gga_lines)
# %%
nmea_sample = load_nmea_file("../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/supplemental/SPAN_Pixel4_10Hz.nmea")
# %%
display(nmea_sample)
# %%
# GNSSLogのテキストをdataframeにしよう
def gnss_log_to_dataframes(path):
    print('Loading ' + path, flush=True)
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
    with open(path) as f_open:
        datalines = f_open.readlines()

    datas = {k: [] for k in gnss_section_names}
    gnss_map = {k: [] for k in gnss_section_names}
    for dataline in datalines:
        is_header = dataline.startswith('#')
        dataline = dataline.strip('#').strip().split(',')
        # skip over notes, version numbers, etc
        if is_header and dataline[0] in gnss_section_names:
            gnss_map[dataline[0]] = dataline[1:]
        elif not is_header:
            datas[dataline[0]].append(dataline[1:])

    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            results[k][col] = pd.to_numeric(results[k][col])
            if col == 'utcTimeMillis' or col == 'UnixTimeMillis':
                results[k]['millisSinceGpsEpoch'] = results[k][col].apply(unix2gps)

    return results
# %%
## 決まった分を日付限定してやっていきましょう
data_path = "../../data/raw/train/2021-03-10-US-SVL-1/Pixel4XL/"
beta_tr_p4 = pd.read_csv(data_path + "Pixel4XL_derived.csv")
beta_gt_p4 = pd.read_csv(data_path + "ground_truth.csv")
# %%
beta_p4_merge = pd.merge_asof(beta_tr_p4, beta_gt_p4, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest")
# %%
beta_p4_merge = beta_p4_merge.drop(["heightAboveWgs84EllipsoidM", "timeSinceFirstFixSeconds", "hDop", "vDop", "speedMps", "courseDegree"], axis=1)
# %%
len(beta_p4_merge["millisSinceGpsEpoch"].unique())
# %%
beta_p4_merge.info()
# %%
dfs = gnss_log_to_dataframes(data_path + 'Pixel4XL_GnssLog.txt')
# %%
for i, df in dfs.items():
    print(i)
    print(df)
# %%
dfs.keys()
# %%
sample_df = pd.read_csv("../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_derived.csv")
# %%
display(sample_df.shape)
# %%
dfs["UncalGyro"].groupby('utcTimeMillis').mean()
# %%
beta_p4_UncalAccel = dfs["UncalAccel"]
# %%
len(beta_p4_UncalAccel["utcTimeMillis"].unique())
# %%
beta_p4_UncalAccel["utcTimeMillis"].plot()
# %%
beta_p4_UncalAccel.info()
# %%
## あっとるんかわからんけどUTCとGPSを変換するらしいコードをパクるよ(PHP製)
from math import fmod
def unix2gps(unix_time):
    if fmod(unix_time, 1) != 0:
        unix_time -= 0.50
        isleap = 1
    else:
        isleap = 0
    gps_time = unix_time - 315964800000
    nleaps = countleaps(gps_time)
    gps_time = gps_time + nleaps + isleap
    return gps_time

def countleaps(gps_time):
    leaps = getleaps()
    lenleaps = len(leaps)
    nleaps = 0
    for i in range(lenleaps):
        if gps_time >= leaps[i] - i:
            nleaps += 1000
    return nleaps

def getleaps():
    leaps = [46828800000, 78364801000, 109900802000, 173059203000, 252028804000, 315187205000, 346723206000, 393984007000, 425520008000, 457056009000, 504489610000, 551750411000, 599184012000, 820108813000, 914803214000, 1025136015000, 1119744016000, 1167264017000]
    return leaps
# %%
beta_p4_UncalAccel["utcTimeMillis"].apply(unix2gps)
# %%
beta_p4_merge["millisSinceGpsEpoch"].head(50)
# %%
beta_p4_UncalAccel["utcTimeMillis"].plot()
# %%
dfs["UncalGyro"]["utcTimeMillis"].plot()
dfs["UncalGyro"].info()
# %%
dfs["UncalMag"]["utcTimeMillis"].plot()
dfs["UncalMag"].info()
# %%
dfs["Status"]["UnixTimeMillis"].plot()
dfs["Status"].info()
# %%
dfs["Raw"]["utcTimeMillis"].plot()
dfs["Raw"].info()
# %%
dfs["Fix"]["UnixTimeMillis"].plot()
# %%
alpha_p4_tr = pd.read_csv("../../data/raw/train/2020-05-14-US-MTV-2/Pixel4/Pixel4_derived.csv")
alpha_p4_gt = pd.read_csv("../../data/raw/train/2020-05-14-US-MTV-2/Pixel4/ground_truth.csv")
# %%
theta_p4_tr = pd.concat([beta_tr_p4, alpha_p4_tr], ignore_index=True)
theta_p4_gt = pd.concat([beta_gt_p4, alpha_p4_gt], ignore_index=True)
theta_p4_merge = pd.merge_asof(theta_p4_tr, theta_p4_gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=100000)
# %%
theta_p4_merge
# %%
## 結合するためのコードを書きます
## とりあえず端末ごとに日毎の取得データとground_truthをくっつけますね
## 端末リストとか共通のパスはハードコーディングで勘弁してくれ
phone_list = ["Pixel4", "Pixel4XL", "Pixel4Modded", "Pixel4XLModded", "Mi8", "SamsungS20Ultra", "Pixel5"]
base_path = "../../data/raw/train/"
dfs = dict()
import os
for root, dirs, files in os.walk(base_path):
    csvs = filter(lambda f: f.endswith(".csv"), files)
    gt = None
    get_data = None
    for csv in csvs:
        if 'Pixel5' not in root:
            continue
        else:
            if gt is None:
                gt = pd.read_csv(os.path.join(root, csv))
            elif get_data is None:
                get_data = pd.read_csv(os.path.join(root, csv))
            if gt is not None and get_data is not None:
                try:
                    merge_data = pd.merge_asof(get_data, gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=100000, allow_exact_matches=True)
                    gnss_data = gnss_log_to_dataframes(os.path.join(root, csv.replace("derived.csv", "GnssLog.txt")))
                    for i, sub_data in gnss_data.items():
                        merge_data = pd.merge_asof(merge_data, sub_data, on="millisSinceGpsEpoch", direction="nearest", tolerance=1000, allow_exact_matches=True, suffixes=['_left', '_right'])
                    dfs[os.path.join(root, csv).split('\\')[1]] = pd.concat([dfs[os.path.join(root, csv).split('\\')[1]], merge_data], ignore_index=True)
                except KeyError:
                    dfs[os.path.join(root, csv).split('\\')[1]] = merge_data
                gt = get_data = None
# %%
dfs.keys()
# %%
dfs["SamsungS20Ultra"]["millisSinceGpsEpoch"].plot()
# %%
dfs
# %%
print(1609881119689 - 315964800000)