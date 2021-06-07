# %%
import optuna.integration.lightgbm as lgb
import pandas as pd

from IPython.core.display import display
from models import Util
#最大表示行数を設定
pd.set_option('display.max_rows', 500)
import warnings
warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

# %%
df_train = pd.read_csv('../../data/interim/train/all_Pixel4_derived.csv')
df_test = pd.read_csv('../../data/interim/test/all_Pixel4_derived.csv')

display(df_train.head())
display(df_test.head())
# %%
