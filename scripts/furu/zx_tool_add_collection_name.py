import pandas as pd
import numpy as np
import glob as gb

# def inputs():
#     return 

# def load_df(path):
#     pd.read_csv()

def main():
    gnss_section_names = {'Raw','UncalAccel', 'UncalGyro', 'UncalMag', 'Fix', 'Status', 'OrientationDeg'}

    path = '../../data/interim/*/*.csv'
    path_list = gb.glob(path)


    paths = []
    for names in gnss_section_names:
        paths += [path for path in path_list if names in path]


    for path in np.array(paths):
        try:
            df = pd.read_csv(path)
        except:
            continue
    
            
    print(len(paths))

if __name__ == '__main__':
    main()