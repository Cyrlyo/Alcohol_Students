import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import Booster
from utilities import importData, prepareData, saveWeights, saveScore

NUM_BOOST_ROUND = 20
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
  return float((1 / (1 + np.exp(-x)))*2)

def XGBoostClassification(data: DataFrame, num_boost_round: int, nfold: int) -> Booster:
    #TODO: delete name
    data_dmatrix = xgb.DMatrix(data=data[data.columns[:-2]].drop(columns=["Dalc", "Walc"]), label=data["alc"], enable_categorical=True)
    xgb_cv = xgb.cv(dtrain=data_dmatrix, params={'objective':'reg:squarederror'}, nfold=nfold, metrics = 'rmse', seed=42, num_boost_round=num_boost_round)
    model = xgb.train({'objective':'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42}, data_dmatrix, num_boost_round=num_boost_round)
    
    return model


def featureImportance(model: Booster) -> dict:
    
    feature_importance = model.get_score(importance_type='gain')
    weights_xgboost = {key:reduceValueRange(value) for key, value in feature_importance.items()}
    weights_xgboost["Name"] = 0
    weights_xgboost["alc"] = 1
    weights_xgboost["Dalc"] = 1
    weights_xgboost["Walc"] = 1

    return weights_xgboost
    
def plotFeatureImportance(model: Booster) -> None:
    
    xgb.plot_importance(model)
    plt.show()

def XGBoostWeightsOptimizer(data: DataFrame) -> dict:
    
    data = changeDataCategory(data)
    model = XGBoostClassification(data, NUM_BOOST_ROUND, NFOLD)
    weights_xgboost = featureImportance(model)
    print("Weights optimized")
    
    saveWeights(weights_xgboost, "./weights", "weights_xgboost.yaml")
    #TODO: add calculate score for a graph
