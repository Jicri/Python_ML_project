import os
import pandas as pd
from feature_engine import *

data_path = os.path.abspath(
    './F24_Flesh_Etch_PD_Process_CDanalysis/data/F24_Flesh_Etch_PD_Process_CDanalysis_train_data_01.csv'
)  #! "relative path"
df = pd.read_csv(data_path)
feature = df[df.columns[:11]]
label = df[df.columns[-2]]
fs = Feature_select(df=df, feature=feature, label=label)
fs.m_rmr()
fs.random_forest(n_estimators=300, max_depth=3, feature=feature, label=label)
fs.xg_regressor(n_estimators=300, max_depth=3, feature=feature, label=label)
123
