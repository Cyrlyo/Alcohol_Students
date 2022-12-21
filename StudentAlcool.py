import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
import community.community_louvain as community_louvain
from mpl_toolkits.mplot3d import Axes3D
import pyvis
import random
from statistics import median
from pandas import DataFrame
from numpy import ndarray
from networkx.classes.graph import Graph

def importData(path: str) -> DataFrame:
    
    data = pd.read_csv(path)
    return data

def prepareData(data: DataFrame) -> DataFrame:
    
    data.rename(columns={"sex":"gender"}, inplace=True)
    
    try: 
        data.drop(columns='Unnamed: 0', inplace=True)
    except: pass
    
    data = createName(data)
    data['alc'] = data['Dalc'] + data['Walc']
    data["guardian"][data["guardian"] == "father"] = "parent"
    data["guardian"][data["guardian"] == "mother"] = "parent"
    data['absences'][(data["absences"] > 0) & (data["absences"] < 11)] = 1

    return data

def createName(data: DataFrame) -> DataFrame:
    
    indexes = [i for i in range(data.shape[0])]
    indexes = pd.DataFrame({"Names":indexes})
    indexes
    data = pd.concat([data, indexes], axis=1)
    
    return data

def printDataInfos(data: DataFrame):
    
    print("\n************************\n")
    print(f"\nShape: {data.shape}\n")
    print(f"Columns:\n{data.columns}")
    print(data.head(3))
    print("\n************************\n")

def DFToNP(data: DataFrame) -> ndarray:
    data_vec = data.to_numpy()
    return data_vec

def printScoresStats(list_of_scores: list):
    
    print("\nDifference score statistics:")
    print(f"Len: {len(list_of_scores)}")
    print(f"Mean: {round(np.mean(list_of_scores), 4)}")
    print(f"Median: {np.median(list_of_scores)}")
    print(f"Max: {np.max(list_of_scores)}")
    print(f"Min: {np.min(list_of_scores)}\n")

def createGraph(G: Graph, data_vec: ndarray) -> Graph:
    
    weights = randomWeights(data)
    
    columns_name = list(data.columns)
    list_of_scores = []
    for vec in range(1, data_vec.shape[0]):
        for vecs in range(data_vec.shape[0]):
            list_difference = []
            score = 0
            for col in range(data_vec.shape[1]):
                if data_vec[vec - 1, col-1] != data_vec[vecs, col-1]:
                    score += weights[columns_name[col]]
                    list_of_scores.append(score)

            if score < 10.5:
                G.add_edge(data_vec[vec, -1], data_vec[vecs, -1])
            else: 
                pass
    printScoresStats(list_of_scores)
    return G

def randomWeights(data: DataFrame) -> dict:
    
    weights = {key:random.uniform(0, 2) for key in list(data.columns)}
    return weights

if __name__ == "__main__":
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    printDataInfos(data)
    
    data_vec = DFToNP(data)
    
    G = nx.Graph()
    createGraph(G, data_vec)