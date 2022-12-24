import os
import pandas as pd
from pandas import DataFrame
import yaml 
from yaml.loader import SafeLoader
from collections import defaultdict
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import ndarray
import argparse

def parseArguments() -> bool:
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--optimize", action="store_true", help="Enable optimizing weights", required=False)
    parser.add_argument("-e", "--epoch", type=int, required=False, help="Number of epoch to optimize weights", default=10, metavar="")
    parser.add_argument("-g", "--graph", action="store_true", help="Create a graph with best saved weights", required=False)
    parser.add_argument("-x", "--xgb_weights", action="store_true", help="Use XGBoost Optimize weights (by default use random optimized weights)")
    
    args = parser.parse_args()
    
    return args.optimize, args.epoch, args.graph, args.xgb_weights
    

def saveScore(score: float, file_name: str = "best_score.txt"):
    
    checkExistingFolder("./weights")
    path_name = os.path.join("./weights", file_name)
    with open(path_name, "w") as file:
        file.write(str(score))

def loadScore(path: str):
    
    with open(path, "r") as file:
        best_score = file.read()
    return best_score

def saveDFToCSV(data: DataFrame):
    
    checkExistingFolder("./Data")    
    data.to_csv("./Data/student_all_community.csv", sep=",", index=False)

def checkExistingFolder(path: str):
    
    result = os.path.exists(path)
    if not result:
        os.mkdir(path)
    
def saveWeights(weights: dict, path: str, file_name: str = "weights.yaml"):
    
    checkExistingFolder(path)
    full_path = os.path.join(path, file_name)
    
    with open(full_path, "w") as file:
        yaml.dump(weights, file, default_flow_style=False)

def loadWeights(path: str) -> dict:
    
    with open(path, "r") as file:
        weights = yaml.load(file, Loader=SafeLoader)
    return weights

def sortedPartition(partition: dict) -> dict:
    
    partition_sorted = dict(sorted(partition.items()))
    return partition_sorted

def refactoringPartition(partition: dict) -> list:

    sets = defaultdict(set)
    for key, value in partition.items():
        sets[value].add(key)
    part_by_com = [sets[x] for x in list(sets.keys())]
    
    return part_by_com

def importData(path: str) -> DataFrame:
    
    data = pd.read_csv(path)
    return data

def prepareData(data: DataFrame) -> DataFrame:
    
    data.rename(columns={"sex":"gender"}, inplace=True)
    
    try: 
        data.drop(columns='Unnamed: 0', inplace=True)
    except: pass
    
    data['alc'] = data['Dalc'] + data['Walc']
    data["guardian"] = data["guardian"].replace({"mother": "parent", "father": "parent"})
    # data["absences"] = data["absences"].apply(discretise)

    data = createName(data)
    return data

def discretise(x: int) -> int:
  if x > 11:
    return x
  elif x > 0:
    return 1
  else:
    return 0

def printScoresStats(list_of_scores: list):
    
    print("\nDifference score statistics:")
    print(f"Len: {len(list_of_scores)}")
    print(f"Mean: {round(np.mean(list_of_scores), 4)}")
    print(f"Median: {round(np.median(list_of_scores), 4)}")
    print(f"Max: {round(np.max(list_of_scores), 4)}")
    print(f"Min: {round(np.min(list_of_scores), 4)}\n")
    
def createName(data: DataFrame) -> DataFrame:
    
    indexes = [i for i in range(data.shape[0])]
    indexes = pd.DataFrame({"Name":indexes})
    data = pd.concat([data, indexes], axis=1)
    
    return data

def addPartitionToData(data: DataFrame, partition: dict) -> DataFrame:
    
    partition_sorted = sortedPartition(partition)
    partition_sorted = pd.DataFrame.from_dict(partition_sorted, orient="index")
    data = pd.concat([data, partition_sorted], axis=1)
    data.rename({0:"Community"}, axis=1, inplace=True)
    
    return data

def plotGraphWithPartition(G: Graph, partition: dict):
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    

def printDataInfos(data: DataFrame):
    
    print("\n************************\n")
    print(f"\nShape: {data.shape}\n")
    print(f"Columns:\n{data.columns}")
    print(data.head(3))
    print("\n************************\n")
    
def graphPlot(G: Graph):
    
    plt.figure(figsize=(12, 8))
    graph_options = {
        'node_color': 'lightblue',
        'node_size' : 10,
        "edge_color": 'grey'
    }

    nx.draw(G, **graph_options, label=True)
    plt.show()
    
def plotGraphStats(G: Graph):
    
    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of selfloops: {len(list(nx.selfloop_edges(G)))}\n")

def DFToNP(data: DataFrame) -> ndarray:
    data_vec = data.to_numpy()
    print(f"\nOriginal shape: {data.shape} | New shape: {data_vec.shape}\n")
    return data_vec