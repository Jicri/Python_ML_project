import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                          colsample_bytree=1,
                          learning_rate=0.01,
                          booster='gbtree',
                          max_depth=2,
                          alpha=1,
                          n_estimators=300,
                          silent=None)
