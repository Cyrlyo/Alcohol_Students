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
from tqdm import tqdm

def importData(path: str) -> DataFrame:
    
    data = pd.read_csv(path)
    return data

def prepareData(data: DataFrame) -> DataFrame:
    
    data.rename(columns={"sex":"gender"}, inplace=True)
    
    try: 
        data.drop(columns='Unnamed: 0', inplace=True)
    except: pass
    
    data['alc'] = data['Dalc'] + data['Walc']
    data["guardian"][data["guardian"] == "father"] = "parent"
    data["guardian"][data["guardian"] == "mother"] = "parent"
    data['absences'][(data["absences"] > 0) & (data["absences"] < 11)] = 1

    data = createName(data)
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
    print(f"\nOriginal shape: {data.shape} | New shape: {data_vec.shape}")
    return data_vec

def printScoresStats(list_of_scores: list):
    
    print("\nDifference score statistics:")
    print(f"Len: {len(list_of_scores)}")
    print(f"Mean: {round(np.mean(list_of_scores), 4)}")
    print(f"Median: {np.median(list_of_scores)}")
    print(f"Max: {np.max(list_of_scores)}")
    print(f"Min: {np.min(list_of_scores)}\n")

def createGraph(G: Graph, data: DataFrame, data_vec: ndarray) -> Graph:
    
    weights = randomWeights(data)
    
    columns_name = list(data.columns)
    list_of_scores = []
    for vec, tq in zip(range(1, data_vec.shape[0]), tqdm(range(1, data_vec.shape[0]))):
        for vecs in range(data_vec.shape[0]):
            list_difference = []
            score = 0
            for col in range(data_vec.shape[1]):
                if data_vec[vec - 1, col-1] != data_vec[vecs, col-1]:
                    score += weights[columns_name[col]]
                    # print(score)
                    list_of_scores.append(score)
# On ne peut pas garder 10.5 il faut trouver un moyen d'avoir une metric qui se calcule. Ou savegarder pour chaque paire
# de noeud le score et en suite y ajouter ou non l'arête
            if score < sum(list(weights.values()))//3:
                G.add_edge(data_vec[vec, -1], data_vec[vecs, -1])
            else: 
                pass
    printScoresStats(list_of_scores)
    plotGraphStats(G)
    return G

def plotGraphStats(G: Graph):
    
    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}\n")

def randomWeights(data: DataFrame) -> dict:
    
    random.seed(42)
    weights = {key:random.uniform(0, 2) for key in list(data.columns)}
    return weights

def graphPlot(G: Graph):
    
    plt.figure(figsize=(12, 8))
    graph_options = {
        'node_color': 'lightblue',
        'node_size' : 10,
        "edge_color": 'grey'
    }

    nx.draw(G, **graph_options, label=True)
    plt.show()

def louvainPartitioning(G: Graph) -> dict:
    
    partition = community_louvain.best_partition(G)
    print(f"\nNumber of partitions: {len(set(partition.values()))}")
    return partition

def plotGraphWithPartition(G: Graph, partition: dict):
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

if __name__ == "__main__":
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    printDataInfos(data)
    
    data_vec = DFToNP(data)
    
    G = nx.Graph()
    G = createGraph(G, data, data_vec)
    graphPlot(G)
    
    partition = louvainPartitioning(G)
    plotGraphWithPartition(G, partition)