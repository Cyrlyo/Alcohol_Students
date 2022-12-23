import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import xgboost as xgb
from utilities import importData, prepareData



if __name__ == "__main__" :
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    
    num_boost_round = 100
    nfold = 5
    
    # data_dmatrix = xgb.DMatrix(data=data[data.columns[:-2]], label=data["alc"], enable_categorical=True)