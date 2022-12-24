import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import community.community_louvain as community_louvain
from mpl_toolkits.mplot3d import Axes3D
import pyvis
import random
from pandas import DataFrame
from numpy import ndarray
from networkx.classes.graph import Graph
from tqdm import tqdm
import os
from typing import List,  Tuple
import time
from utilities import *

DATA_PATH = "./Data/student_all.csv"

def createGraph(G: Graph, data: DataFrame, data_vec: ndarray, reuse: bool = True, random_weights: bool = True) -> Tuple[Graph, dict]:
    
    if random_weights:
        weights = randomWeights(data, reuse)
    else:
        weights = loadWeights("./weights/weights_random.yaml")
    
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

def randomWeights(data: DataFrame, reuse: bool) -> dict:
    
    if reuse:
        random.seed(42)
    weights = {key:random.uniform(0, 2) for key in list(data.columns)}
    weights["Name"] = 0
    weights["Alc"] = 1
    weights["Dalc"] = 1
    weights["Walc"] = 1
    
    return weights

def louvainPartitioning(G: Graph) -> dict:
    
    partition = community_louvain.best_partition(G)
    print(f"\nNumber of partitions: {len(set(partition.values()))}")
    return partition

def louvain_community_quality(G: Graph, communities: list[set]) -> float:
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

def findBestRandomWeight(data: DataFrame, data_vec: ndarray, epoch: int=10) -> Tuple[dict, ndarray]:
    
    results = {}
    score_list = []
    
    for i in range(epoch):
        start_time = time.time()
        print("----------------")
        
        try:
            best_saved_score = float(loadScore("./weights/best_score_random.txt"))
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
            if not os.path.exists("./weights/best_score_random.txt") or not os.path.exists("./weights/weights_random.yaml"):
                saveWeights(weights, "./weights", "weights_random.yaml")
                saveScore(max(score_list), "best_score_random.txt")
                print("\nWeights & score saved")
    
        delta_time = time.time() - start_time
        print(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(delta_time))}")
    return results, np.array(score_list)

def retrieveData(path: str) -> Tuple[DataFrame, ndarray]:
    
    data = importData(path)
    data = prepareData(data)
    printDataInfos(data)
    
    data_vec = DFToNP(data)
    
    return data, data_vec

def randomWeightOptimizer(data: DataFrame, data_vec: ndarray, epoch: int = 10):
    
    results, score_list = findBestRandomWeight(data, data_vec, epoch)
    print("--------------------------------------------")
    print(f"\nBest model: model_{np.argmax(score_list)}")
    print(f"Best score: {max(score_list)}")


def count_and_remove_extra_edges(graph):
  # On parcourt tous les noeuds du graphe
  count = 0
  for node1 in graph.nodes():
    for node2 in graph.nodes():
      # On ne s'intéresse qu'aux noeuds différents
      if node1 != node2:
        # On compte le nombre d'arêtes entre les deux noeuds
        edge_count = len(graph.edges(node1, node2))

        # Si il y a plus d'une arête, on les supprime toutes
        if edge_count > 1:
        #   graph.remove_edges_from(graph.edges(node1, node2))
            count += 1
    print(count)

if __name__ == "__main__":
    
    start_time = time.time()
    
    optimize, epoch, graph = parseArguments()
    
    
    data, data_vec = retrieveData(DATA_PATH)
    
    if optimize:
        randomWeightOptimizer(data, data_vec, epoch)
# Change weights_random.yaml name by random_weights_random.yaml
# same for best_score_random.txt
    
    
    if graph:
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