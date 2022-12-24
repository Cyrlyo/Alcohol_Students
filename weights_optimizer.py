import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import Booster
from utilities import importData, prepareData

NUM_BOOST_ROUND = 100
NFOLD = 5


def changeDataCategory(data: DataFrame) -> DataFrame:
    
    unique_values = dict()
    # print(f"colonne\tnb_unique_values\tunique_values_sorted")
    for col, col_type in data.dtypes.items():
        if col_type == 'object' and col != 'G1' and col != 'G2' and col != 'G3' and col != 'absences':
            data[col] = data[col].astype('category')
            unique_values[col] = np.sort(np.unique(data[col]))
            # print(f"{col}:\t{len(unique_values[col])}\t{unique_values[col]}\n")
    return data

def reduceValueRange(x: float) -> float:
    #TODO: rename this function
  return (1 / (1 + np.exp(-x)))*2

def XGBoostClassification(data: DataFrame, num_boost_round: int, nfold: int) -> Booster:
    
    data_dmatrix = xgb.DMatrix(data=data[data.columns[:-2]].drop(columns=["Dalc", "Walc"]), label=data["alc"], enable_categorical=True)
    xgb_cv = xgb.cv(dtrain=data_dmatrix, params={'objective':'reg:squarederror'}, nfold=nfold, metrics = 'rmse', seed=42, num_boost_round=num_boost_round)
    model = xgb.train({'objective':'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}, data_dmatrix, num_boost_round=num_boost_round)
    
    return model


def featureImportance(model: Booster) -> dict:
    
    feature_importance = model.get_score(importance_type='gain')
    weights_xgboost = {key:reduceValueRange(value) for key, value in feature_importance.items()}

    return weights_xgboost
    
def plotFeatureImportance(model: Booster) -> None:
    
    xgb.plot_importance(model)
    plt.show()

if __name__ == "__main__" :
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    data = changeDataCategory(data)


    # plt.figure(figsize=(14, 8))
    # plt.errorbar(xgb_cv.index, xgb_cv['train-rmse-mean'], yerr=xgb_cv['train-rmse-std'], label='train_rmse')
    # plt.errorbar(xgb_cv.index, xgb_cv['test-rmse-mean'], yerr=xgb_cv['test-rmse-std'], label='valid_rmse')
    # plt.legend()
    # plt.title(f'Evolution of the train and validation mean and standard deviation RMSE over {nfold} cross-validation folds as a function of the round number')
    # plt.xlabel('Round number')
    # plt.ylabel('Mean RMSE')
    # plt.show()
    