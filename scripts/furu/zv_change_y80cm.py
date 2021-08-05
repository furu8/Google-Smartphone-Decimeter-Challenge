import pandas as pd
import numpy as np

def evaluate_lat_lng_dist(df):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Radius of earth in kilometers is 6367 or 6371
    RADIUS = 6371000
    # RADIUS = 6367000
    
    dist_list = []

    for i in range(len(df)):
        lat_truth = df.loc[i, 'lat_truth']
        lng_truth = df.loc[i, 'lng_truth']
        lat_pred = df.loc[i, 'lat_pred']
        lng_pred = df.loc[i, 'lng_pred']
        # convert decimal degrees to radians 
        lng_truth, lat_truth, lng_pred, lat_pred = map(np.deg2rad, [lng_truth, lat_truth, lng_pred, lat_pred])
        # haversine formula 
        dlng = lng_pred - lng_truth 
        dlat = lat_pred - lat_truth 
        a = np.sin(dlat/2)**2 + np.cos(lat_truth) * np.cos(lat_pred) * np.sin(dlng/2)**2
        dist = 2 * RADIUS * np.arcsin(np.sqrt(a))
        dist_list.append(dist)

    return dist_list

df = pd.read_csv('../../data/submission/submission4.csv')
df['latDeg'] = df['latDeg'] + 0.0000088
# df['lngDeg'] = df['lngDeg'] + 0.0000088

df.to_csv('../../data/submission//submission4_+lat.csv', index=False)