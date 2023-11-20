from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from mrmr import *

import pandas as pd
import numpy as np
import time
import os


class Feature_select():

    def __init__(self, df, feature, label):
        self.df = df
        self.feature = feature
        self.label = label

    def m_rmr(self):
        mrmr_importances = mrmr_regression(self.feature, self.label, K=10)
        feature_number = [i + 1 for i in range(10)]
        mrmr_score = {i: j for i, j in zip(mrmr_importances, feature_number)}
        df = pd.DataFrame.from_dict(mrmr_score, orient="index")
        self.create_table(df, "mRMR")

    def random_forest(self, n_estimators, max_depth, feature, label):
        forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model = forest.fit(self.feature, self.label)
        forest_importances_index = model.feature_importances_.argsort()[::-1][:10]
        forest_importance_score = model.feature_importances_[forest_importances_index]
        feature_name = feature.columns[forest_importances_index]
        forest_score = {i: j for i, j in zip(feature_name, forest_importance_score)}
        df = pd.DataFrame.from_dict(forest_score, orient="index")
        self.create_table(df, "random_forest")

    def xg_regressor(self, n_estimators, max_depth, feature, label):
        xg_reg = XGBRegressor(objective='reg:squarederror',
                              learning_rate=0.01,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              colsample_bytree=0.1)
        model = xg_reg.fit(self.feature, self.label)
        xgboost_importances_index = model.feature_importances_.argsort()[::-1][:10]
        xgboost_importances_score = model.feature_importances_[xgboost_importances_index]
        feature_name = feature.columns[xgboost_importances_index]
        xg_score = {i: j for i, j in zip(feature_name, xgboost_importances_score)}
        df = pd.DataFrame.from_dict(xg_score, orient="index")
        self.create_table(df, "xg_classifier")
        return xg_score

    def create_table(self, df, name):
        localtime = time.strftime("%H%M%S")
        localdate = time.strftime("%Y%m%d")
        path = os.path.abspath('./F24_Flesh_Etch_PD_Process_CDanalysis/data/feature_engineer/{}/{}={}, {}.xlsx'.format(
            name, name, localdate, localtime))
        df.to_excel(path)
