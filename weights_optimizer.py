import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from utilities import importData, prepareData


def changeDataCategory(data: DataFrame) -> DataFrame:
    
    unique_values = dict()
    # print(f"colonne\tnb_unique_values\tunique_values_sorted")
    for col, col_type in data.dtypes.items():
        if col_type == 'object' and col != 'G1' and col != 'G2' and col != 'G3' and col != 'absences':
            data[col] = data[col].astype('category')
            unique_values[col] = np.sort(np.unique(data[col]))
            # print(f"{col}:\t{len(unique_values[col])}\t{unique_values[col]}\n")
    return data

if __name__ == "__main__" :
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    data = changeDataCategory(data)
    
    num_boost_round = 100
    nfold = 5
    
    data_dmatrix = xgb.DMatrix(data=data[data.columns[:-2]], label=data["alc"], enable_categorical=True)
    xgb_cv = xgb.cv(dtrain=data_dmatrix, params={'objective':'reg:squarederror'}, nfold=nfold, metrics = 'rmse', seed=42, num_boost_round=num_boost_round)

    # Train the model
    num_boost_round = 20
    clf = xgb.train({'objective':'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}, data_dmatrix, num_boost_round=num_boost_round)

    # plt.figure(figsize=(14, 8))
    # plt.errorbar(xgb_cv.index, xgb_cv['train-rmse-mean'], yerr=xgb_cv['train-rmse-std'], label='train_rmse')
    # plt.errorbar(xgb_cv.index, xgb_cv['test-rmse-mean'], yerr=xgb_cv['test-rmse-std'], label='valid_rmse')
    # plt.legend()
    # plt.title(f'Evolution of the train and validation mean and standard deviation RMSE over {nfold} cross-validation folds as a function of the round number')
    # plt.xlabel('Round number')
    # plt.ylabel('Mean RMSE')
    # plt.show()

    print(clf.get_score(importance_type='gain'))


    # Plot feature importance
    xgb.plot_importance(clf)
    # plt.figure(figsize = (16, 12))
    plt.show()