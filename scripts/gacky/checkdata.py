# %%
## 各種インポート
from re import sub
import numpy as np
from numpy.lib.function_base import disp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, display_pdf
import glob as gb
import matplotlib.pyplot as plt
import seaborn as sns
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
            try:
                datas[dataline[0]].append(dataline[1:])
            except KeyError:
                print(dataline[0])

    results = dict()
    for k, v in datas.items():
        results[k] = pd.DataFrame(v, columns=gnss_map[k])
    # pandas doesn't properly infer types from these lists by default
    for k, df in results.items():
        for col in df.columns:
            if col == 'CodeType':
                continue
            try:
                results[k][col] = pd.to_numeric(results[k][col])
                if col == 'utcTimeMillis' or col == 'UnixTimeMillis':
                    results[k]['millisSinceGpsEpoch'] = results[k][col].apply(unix2gps)
                    results[k].drop(col, axis=1, inplace=True)
            except ValueError:
                results[k] = results[k][:-1]
                results[k][col] = pd.to_numeric(results[k][col])
    #results["Raw"]["rawPrM"] = results["Raw"]["TimeNanos"] - results["Raw"]["FullBiasNanos"] - results["Raw"]["BiasNanos"]

    return results
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
## GnssLogのデータフレームたちを一つにまとめます
def merge_gnss_data(gnss_data):
    gnss_section_names = ["Raw", "Status", "UncalAccel", "UncalGyro", "UncalMag", "Fix", "OrientationDeg"]
    gnss_data["Raw"]["elapsedRealtimeNanos"] = np.nan
    for sub_data in gnss_section_names:
        try:
            merge_data = pd.merge_asof(merge_data, gnss_data[sub_data], on="millisSinceGpsEpoch", direction="nearest", tolerance=1000, allow_exact_matches=True, suffixes=('', sub_data))
        except UnboundLocalError:
            merge_data = gnss_data[sub_data]
        except KeyError:
            print(f'{sub_data} is not in this log data.')
    merge_data.drop("elapsedRealtimeNanos", axis=1, inplace=True)
    #display(merge_data)
    return merge_data
# %%
## ソート問題を何とかする関数
def sort_gnss_dataframe(gnss_dict):
    for sub_name in gnss_dict.keys():
        try:
            gnss_dict[sub_name]["Index"] = gnss_dict[sub_name].index
            gnss_dict[sub_name] = gnss_dict[sub_name].sort_values(["millisSinceGpsEpoch", "Index"], ignore_index=True)
            gnss_dict[sub_name].drop("Index", axis=1, inplace=True)
        except KeyError:
            print(f"{sub_name} is not have millisSinceGpsEpoch")
    return gnss_dict
# %%
df_dict = gnss_log_to_dataframes("../../data/raw/train/2021-01-05-US-SVL-1/Pixel5/Pixel5_GnssLog.txt")
derived_data = pd.read_csv("../../data/raw/train/2021-01-05-US-SVL-1/Pixel5/Pixel5_derived.csv")
gt = pd.read_csv("../../data/raw/train/2021-01-05-US-SVL-1/Pixel5/ground_truth.csv")
# %%
df_dict['Raw'].info()
df_dict['Status'].info()
len(set(df_dict['Raw']["millisSinceGpsEpoch"].unique()) - set(df_dict['Status']["millisSinceGpsEpoch"].unique()))
# %%
dfs = merge_gnss_data(df_dict)
display(dfs.info())
merge_data = pd.merge_asof(pd.merge_asof(dfs, derived_data, on="millisSinceGpsEpoch", direction="nearest", tolerance=1000, allow_exact_matches=True), gt, on="millisSinceGpsEpoch", direction="nearest", tolerance=1000, allow_exact_matches=True)
merge_data.describe()
# %%
for i in merge_data.columns:
    print(i)

# %%
for i, value in merge_data.isnull().sum().iteritems():
    if value > 0:
        print(i, value)
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
        if 'Mi8' not in root:
            continue
        else:
            if gt is None:
                gt = pd.read_csv(os.path.join(root, csv))
            elif get_data is None:
                get_data = pd.read_csv(os.path.join(root, csv))
                get_data.rename(columns={"svid": "Svid", "constellationType": "ConstellationType"}, inplace=True)
            if gt is not None and get_data is not None:
                try:
                    gnss_data = gnss_log_to_dataframes(os.path.join(root, csv).replace("derived.csv", "GnssLog.txt"))
                    merge_data = pd.merge_asof(pd.merge_asof(merge_gnss_data(sort_gnss_dataframe(gnss_data)), get_data, on="millisSinceGpsEpoch", direction="nearest", tolerance=100000, allow_exact_matches=True), gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=100000, allow_exact_matches=True)
                    #merge_data_2 = pd.merge_asof(merge_gnss_data(sort_gnss_dataframe(gnss_data)), get_data, on="millisSinceGpsEpoch", by=["Svid", "ConstellationType"], direction="nearest", tolerance=100000, allow_exact_matches=True)
                    #merge_data = pd.merge_asof(merge_data_2, gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=100000, allow_exact_matches=True)
                    dfs[os.path.join(root, csv).split('\\')[1]] = pd.concat([dfs[os.path.join(root, csv).split('\\')[1]], merge_data])
                    #dfs[os.path.join(root, csv).split('\\')[1] + "_2"] = pd.concat([dfs[os.path.join(root, csv).split('\\')[1] + "_2"], merge_data_2])
                except KeyError:
                    dfs[os.path.join(root, csv).split('\\')[1]] = merge_data
                    #dfs[os.path.join(root, csv).split('\\')[1] + "_2"] = merge_data_2
            #except pd.errors.InvalidIndexError:
            #    display(dfs[os.path.join(root, csv).split('\\')[1]])
            #    display(merge_data)
                gt = get_data = None
# %%
dfs["Pixel4"].to_csv("../../data/interim/merged_test_Pixel4.csv", sep=",")
dfs["Pixel4XL"].to_csv("../../data/interim/merged_test_Pixel4XL.csv", sep=",")
dfs["Pixel4Modded"].to_csv("../../data/interim/merged_test_Pixel4Modded.csv", sep=",")
dfs["Pixel4XLModded"].to_csv("../../data/interim/merged_test_Pixel4XLModded.csv", sep=",")
dfs["Pixel5"].to_csv("../../data/interim/merged_test_Pixel5.csv", sep=",")
dfs["Mi8"].to_csv("../../data/interim/merged_test_Mi8.csv", sep=",")
dfs["SamsungS20Ultra"].to_csv("../../data/interim/merged_test_SamsungS20Ultra.csv", sep=",")
# %%
for col in dfs["Mi8_2"].columns:
    print(col)
# %%
dfs["Mi8_2"]["collectionName"]#[dfs["Mi8"]["collectionName"] == "2021-01-15-US-SVL-1"]
# %%
def check_sort(gnss_dict):
    for sub_name, gnss_datum in gnss_dict.items():
        #print(sub_name, gnss_datum)
        #for col in gnss_datum.columns:
        try:
            if len(gnss_dict[sub_name][gnss_dict[sub_name]["millisSinceGpsEpoch"].diff()<0]) > 0:
                print(sub_name)
                display(gnss_dict[sub_name][gnss_dict[sub_name]["millisSinceGpsEpoch"].diff()<0])
        except KeyError:
            print(f'{sub_name} is not have millisSinceGpsEpoch')
# %%
## なんか順番おかしいのがあるらしいんで探そ
import os
phone_list = ["Pixel4", "Pixel4XL", "Pixel4Modded", "Pixel4XLModded", "Mi8", "SamsungS20Ultra", "Pixel5"]
base_path = "../../data/raw/train/"
df = pd.DataFrame
for root, dirs, files in os.walk(base_path):
    gnss_logs = filter(lambda f: f.endswith("Pixel4_GnssLog.txt"), files)
    #gt = None
    #get_data = None
    for gnss_log in gnss_logs:
        print(os.path.join(root, gnss_log))
        #if gt is None:
        #    gt = pd.read_csv(os.path.join(root, csv))
        #elif get_data is None:
        #    get_data = pd.read_csv(os.path.join(root, csv))
        #if gt is not None and get_data is not None:
        #gnss_dict = gnss_log_to_dataframes(os.path.join(root, gnss_log))
        #check_sort(gnss_dict)
# %%
gnss_dict = gnss_log_to_dataframes("../../data/raw/train/2020-07-17-US-MTV-2/Mi8/Mi8_GnssLog.txt")
# %%
check_sort(gnss_dict)
# %%
len(gnss_dict['UncalGyro'][3800:3802])
# %%
## Pixel4の結合をなんとかしなきゃ
gnss_dict_Pixel4_1 = gnss_log_to_dataframes("../../data/raw/train/2020-05-14-US-MTV-1/Pixel4/Pixel4_GnssLog.txt")
# %%
check_sort(gnss_dict_Pixel4_1)
# %%
gnss_dict_Pixel4_1["Status"]["Index"] = gnss_dict_Pixel4_1["Status"].index
gnss_dict_Pixel4_1["Status"] = gnss_dict_Pixel4_1["Status"].sort_values(["millisSinceGpsEpoch", "Index"], ignore_index=True)
check_sort(gnss_dict_Pixel4_1)
# %%
gnss_dict_Pixel4_1["Status"].drop("Index", axis=1, inplace=True)
# %%
for col in gnss_dict_Pixel4_1["Raw"].columns:
    print(col)
# %%
gnss_dict_Pixel4_1["Raw"][["Svid", "ConstellationType", "millisSinceGpsEpoch"]]
# %%
gnss_dict_Pixel4_1["UncalGyro"]["Index"] = gnss_dict_Pixel4_1["UncalGyro"].index
gnss_dict_Pixel4_1["UncalGyro"] = gnss_dict_Pixel4_1["UncalGyro"].sort_values(["millisSinceGpsEpoch", "Index"], ignore_index=True)
gnss_dict_Pixel4_1["UncalAccel"]["Index"] = gnss_dict_Pixel4_1["UncalAccel"].index
gnss_dict_Pixel4_1["UncalAccel"] = gnss_dict_Pixel4_1["UncalAccel"].sort_values(["millisSinceGpsEpoch", "Index"], ignore_index=True)
check_sort(gnss_dict_Pixel4_1)
# %%
gnss_dict_Pixel4_1["UncalAccel"][68305:68315]
# %%
Mi8_df = pd.read_csv('../../data/interim/merged_Mi8.csv', index_col=0)
# %%
Mi8_df.head()
# %%
def check_df_sort(gnss_df):
    if len(gnss_df[gnss_df["millisSinceGpsEpoch"].diff()<0]) > 0:
        display(gnss_df[gnss_df["millisSinceGpsEpoch"].diff()<0])
    else:
        print("No Problem.")

# %%
check_df_sort(Mi8_df)
# %%
Pixel4_df = pd.read_csv('../../data/interim/merged_Pixel4.csv', index_col=0)
# %%
check_df_sort(Pixel4_df)
# %%
Pixel4_df[Pixel4_df["collectionName"] == "2021-04-29-US-SJC-2"][["receivedSvTimeInGpsNanos", "millisSinceGpsEpoch"]].unique()

# %%
Pixel4_df["SignalIndex"]
# %%
phone_list = ["Pixel4", "Pixel4XL", "Pixel4Modded", "Pixel4XLModded", "Mi8", "SamsungS20Ultra", "Pixel5"]
base_path = "../../data/raw/test/"
dfs = dict()
import os
for root, dirs, files in os.walk(base_path):
    csvs = filter(lambda f: f.endswith(".csv"), files)
    get_data = None
    for csv in csvs:
        if get_data is None:
            get_data = pd.read_csv(os.path.join(root, csv))
            get_data.rename(columns={"svid": "Svid", "constellationType": "ConstellationType"}, inplace=True)
        if gt is not None and get_data is not None:
            try:
                gnss_data = gnss_log_to_dataframes(os.path.join(root, csv).replace("derived.csv", "GnssLog.txt"))
                merge_data = pd.merge_asof(pd.merge_asof(merge_gnss_data(sort_gnss_dataframe(gnss_data)), get_data, on="millisSinceGpsEpoch", direction="nearest", tolerance=100000, allow_exact_matches=True), gt, on="millisSinceGpsEpoch", by=["collectionName", "phoneName"], direction="nearest", tolerance=100000, allow_exact_matches=True)
                dfs[os.path.join(root, csv).split('\\')[1]] = pd.concat([dfs[os.path.join(root, csv).split('\\')[1]], merge_data])
            except KeyError:
                dfs[os.path.join(root, csv).split('\\')[1]] = merge_data
            get_data = None
# %%
s = -7.0277543E-4
print(f"{s:f}")
# %%
gnss_log_to_dataframes("../../data/raw/test/2021-04-21-US-MTV-1/Pixel4Modded/Pixel4Modded_GnssLog.txt")

# %%
## gnssをそれぞれでconcatとする
def concat_gnss_df(base, new):
    for key in base.keys():
        try:
            base[key] = pd.concat([base[key], new[key]], ignore_index=True)
        except TypeError:
            base[key] = new[key]
    return base
# %%
## これでGNSSを一気に吐き出せるはず
## 結合するためのコードを書きます
## とりあえず端末ごとに日毎の取得データとground_truthをくっつけますね
## 端末リストとか共通のパスはハードコーディングで勘弁してくれ
phone_list = ["Pixel4", "Pixel4XL", "Pixel4Modded", "Pixel4XLModded", "Mi8", "SamsungS20Ultra", "Pixel5"]
base_path = "../../data/raw/"
#gnss_df = pd.DataFrame()
gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}
dfs = {k: {l: [] for l in gnss_section_names} for k in phone_list}
#gnss_df = {k: [] for k in gnss_section_names}
#phone = "Pixel4XLModded"
#gt = pd.DataFrame()
#derived_data = pd.DataFrame()
import os
for path in ['train', 'test']:
    for root, dirs, files in os.walk(base_path + path):
        #csvs = filter(lambda f: f.endswith(".csv"), files)
        txts = filter(lambda f: f.endswith(".txt"), files)
        for txt in txts:
            #print(txt)
            #if phone not in root:
            #    continue
            #else:
            #if 'ground_truth' in csv:
                #gt = pd.concat([gt, pd.read_csv(os.path.join(root, csv))], ignore_index=True)
            #elif 'derived' in csv:
                #derived_data = pd.concat([derived_data, pd.read_csv(os.path.join(root, csv))], ignore_index=True)
            #try:
            gnss_data = gnss_log_to_dataframes(os.path.join(root, txt))#.replace("derived.csv", "GnssLog.txt"))
            #merge_data = merge_gnss_data(sort_gnss_dataframe(gnss_data))
            #gnss_df = pd.concat([gnss_df, merge_data], ignore_index=True)
            dfs[txt.split('_')[0]] = concat_gnss_df(dfs[txt.split('_')[0]], gnss_data)
            #except KeyError:
                #gnss_df = merge_data
                #gnss_df = gnss_data
            #gt = get_data = None
    #display(dfs['Mi8'])
    for key in dfs.keys():
        for sub_key in dfs[key].keys():
            dfs[key][sub_key].to_csv(f'../../data/interim/{path}/merged_{key}_{sub_key}.csv', sep=',', index=False)
#derived_data.rename(columns={"svid": "Svid", "constellationType": "ConstellationType"}, inplace=True)
#gt.to_csv(f'../../data/interim/test/merged_{phone}_gt.csv', sep=',', index=False)
#derived_data.to_csv(f'../../data/interim/test/merged_{phone}_derived.csv', sep=',', index=False)
#gnss_df.to_csv(f'../../data/interim/test/merged_{phone}_gnss.csv', sep=',', index=False)

# %%
gt[gt["millisSinceGpsEpoch"].diff()<0]
# %%
derived_data[derived_data["millisSinceGpsEpoch"].diff()<0]
# %%
derived_data
# %%
#gnss_df[gnss_df["millisSinceGpsEpoch"].diff()<0]
# %%
gt.to_csv('../../data/interim/train/merged_Mi8_gt.csv', sep=',', index=False)
derived_data.to_csv('../../data/interim/train/merged_Mi8_derived.csv', sep=',', index=False)
#gnss_df.to_csv('../../data/interim/train/merged_Mi8_gnss.csv', sep=',', index=False)
# %%
