import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from utilities import importData, prepareData



if __name__ == "__main__" :
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    
    print(data.columns[:-1])
    print(data.dtypes)
    
    num_boost_round = 100
    nfold = 5
    
    if True:
        data_dmatrix = xgb.DMatrix(data=data[data.columns[:-1]], label=data["alc"], enable_categorical=True)
        xgb_cv = xgb.cv(dtrain=data_dmatrix, params={'objective':'reg:squarederror'}, nfold=nfold, metrics = 'rmse', seed=42, num_boost_round=num_boost_round)

        # Train the model
        num_boost_round = 20
        clf = xgb.train({'objective':'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}, data_dmatrix, num_boost_round=num_boost_round)

        # Plot feature importance
        xgb.plot_importance(clf)
        plt.figure(figsize = (16, 12))
        plt.show()