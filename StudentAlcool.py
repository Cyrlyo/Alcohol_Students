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
from collections import defaultdict
import os
from typing import List,  Tuple
import yaml
from yaml.loader import SafeLoader
import time
import argparse

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
    indexes = pd.DataFrame({"Name":indexes})
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
    print(f"\nOriginal shape: {data.shape} | New shape: {data_vec.shape}\n")
    return data_vec

def printScoresStats(list_of_scores: list):
    
    print("\nDifference score statistics:")
    print(f"Len: {len(list_of_scores)}")
    print(f"Mean: {round(np.mean(list_of_scores), 4)}")
    print(f"Median: {round(np.median(list_of_scores), 4)}")
    print(f"Max: {round(np.max(list_of_scores), 4)}")
    print(f"Min: {round(np.min(list_of_scores), 4)}\n")

def createGraph(G: Graph, data: DataFrame, data_vec: ndarray, reuse: bool = True, random_weights: bool = True) -> Tuple[Graph, dict]:
    
    if random_weights:
        weights = randomWeights(data, reuse)
    else:
        weights = loadWeights("./weights/weights.yaml")
    
    columns_name = list(data.columns)
    list_of_scores = []
    for vec, tq in zip(range(1, data_vec.shape[0]), tqdm(range(1, data_vec.shape[0]))):
        for vecs in range(data_vec.shape[0]):
            score = 0
            for col in range(data_vec.shape[1]):
                if data_vec[vec - 1, col-1] != data_vec[vecs, col-1]:
                    score += weights[columns_name[col]]
                    list_of_scores.append(score)

            if score < sum(list(weights.values()))//3:
                G.add_edge(data_vec[vec, -1], data_vec[vecs, -1])
            else: 
                pass

    printScoresStats(list_of_scores)
    plotGraphStats(G)
    G = prepareGraph(G)
    return G, weights

def prepareGraph(G: Graph) -> Graph:
    
    print("Deleting selfloops\n")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def plotGraphStats(G: Graph):
    
    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of selfloops: {len(list(nx.selfloop_edges(G)))}\n")

def randomWeights(data: DataFrame, reuse: bool) -> dict:
    
    if reuse:
        random.seed(42)
    weights = {key:random.uniform(0, 2) for key in list(data.columns)}
    weights["Name"] = 0
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

def louvain_community_quality(G, communities):
    """
    Calculate the quality of a Louvain community detection using the modularity score.
    
    Parameters:
    - G: NetworkX graph
    - communities: list of sets, representing the detected communities
    
    Returns:
    - modularity: float, the modularity score
    """
    modularity = nx.algorithms.community.modularity(G, communities)
    print(f"Modularity for this graph: {round(modularity, 4)}\n")
    
    return modularity

def refactoringPartition(partition: dict) -> list:

    sets = defaultdict(set)
    for key, value in partition.items():
        sets[value].add(key)
    part_by_com = [sets[x] for x in list(sets.keys())]
    
    return part_by_com

def sortedPartition(partition: dict) -> dict:
    
    partition_sorted = dict(sorted(partition.items()))
    return partition_sorted

def addPartitionToData(data: DataFrame, partition: dict) -> DataFrame:
    
    partition_sorted = sortedPartition(partition)
    partition_sorted = pd.DataFrame.from_dict(partition_sorted, orient="index")
    data = pd.concat([data, partition_sorted], axis=1)
    data.rename({0:"Community"}, axis=1, inplace=True)
    
    return data

def saveDFToCSV(data: DataFrame):
    
    checkExistingFolder("./Data")    
    data.to_csv("./Data/student_all_community.csv", sep=",", index=False)

def checkExistingFolder(path: str):
    
    result = os.path.exists(path)
    if not result:
        os.mkdir(path)
    
def saveWeights(weights: dict, path: str):
    
    checkExistingFolder(path)
    full_path = os.path.join(path, "weights.yaml")
    
    with open(full_path, "w") as file:
        yaml.dump(weights, file, default_flow_style=False)

def loadWeights(path: str) -> dict:
    
    with open(path, "r") as file:
        weights = yaml.load(file, Loader=SafeLoader)
    return weights

def findBestRandomWeight(data: DataFrame, data_vec: ndarray) -> Tuple[dict, ndarray]:
    
    results = {}
    score_list = []
    
    for i in range(1):
        start_time = time.time()
        print("----------------")
        
        try:
            best_saved_score = float(loadScore("./weights/best_score.txt"))
            print(f"Best saved score: {best_saved_score}")
        except:
            print("Saved score not found, will be created")
        
        print(f"\nEpoch: {i}")
        G = nx.Graph()
        G, weights = createGraph(G, data, data_vec, reuse=False)
        
        partition = louvainPartitioning(G)
        part_by_com = refactoringPartition(partition)
        modularity = louvain_community_quality(G, part_by_com)
        
        results[f"model_{i}"] = {"weights": weights}
        score_list.append(modularity)
        print(f"\nScore: {modularity}\n\n")
        
        
        try:
            if max(score_list) > best_saved_score:
                saveWeights(weights, "./weights")
                saveScore(max(score_list))
                print("\nWeights & score saved")
        except: 
            if not os.path.exists("./weights/best_score.txt") or not os.path.exists("./weights/weights.yaml"):
                saveWeights(weights, "./weights")
                saveScore(max(score_list))
                print("\nWeights & score saved")
    
        delta_time = time.time() - start_time
        print(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(delta_time))}")
    return results, np.array(score_list)

def saveScore(score: float):
    
    checkExistingFolder("./weights")
    with open("./weights/best_score.txt", "w") as file:
        file.write(str(score))

def loadScore(path: str):
    
    with open("./weights/best_score.txt", "r") as file:
        best_score = file.read()
    return best_score

if __name__ == "__main__":
    
    start_time = time.time()
    
    data = importData("./Data/student_all.csv")
    data = prepareData(data)
    printDataInfos(data)
    
    data_vec = DFToNP(data)
    
    if True:
        results, score_list = findBestRandomWeight(data, data_vec)
        print("--------------------------------------------")
        print(f"\nBest model: model_{np.argmax(score_list)}")
        best_weights = results["model_%s"% np.argmax(score_list)]["weights"]
        print(f"Best score: {max(score_list)}")
    
    
    if False:
        G = nx.Graph()
        G, weights = createGraph(G, data, data_vec, random_weights=False)
        graphPlot(G)

        partition = louvainPartitioning(G)
        plotGraphWithPartition(G, partition)

        part_by_com = refactoringPartition(partition)
        modularity = louvain_community_quality(G, part_by_com)

        data = addPartitionToData(data, partition)
        saveDFToCSV(data)
        
    delta_time = time.time() - start_time
    print(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(delta_time))}")