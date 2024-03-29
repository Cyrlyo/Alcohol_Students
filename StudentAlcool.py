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
from weights_optimizer import XGBoostWeightsOptimizer

DATA_PATH = "./Data/student_all.csv"

def createGraph(G: Graph, data: DataFrame, data_vec: ndarray, reuse: bool = True, random_weights: bool = True, xbgweights: bool = False) -> Tuple[Graph, dict]:
    """
    
    This function takes in a Graph object, a DataFrame, and an ndarray as input. It also takes in optional boolean
    arguments for reusing weights, using random weights, and using xgboost weights. 

    The function calculates the weights for the edges of the graph based on the input data and weights.
    It then adds edges to the graph based on a threshold score calculated from the weights and the differences
    in the data vectors. The function also includes print and plot functions for displaying statistics about
    the graph and the scores.
    
    Parameters:
        - G (Graph): Graph object to update
        - data (DataFrame): Input data for calculating weights
        - data_vec (ndarray): Array of data vectors for adding edges to the graph
        - reuse (bool): Optional argument for reusing weights. Default is True
        - random_weights (bool): Optional argument for using random weights. Default is True
        - xbgweights (bool): Optional argument for using xgboost weights. Default is False

    Returns:
        Tuple[Graph, dict]: Updated Graph object and the weights used for the edges

    """
    
    if random_weights:
        weights = randomWeights(data, reuse)
    else:
        if xbgweights:
            weights = loadWeights("./weights/weights_xgboost.yaml")
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
                #TODO: on pourrait optimiser le seuil (le diviser par 3)
                G.add_edge(data_vec[vec, -1], data_vec[vecs, -1])
            else: 
                pass

    printScoresStats(list_of_scores)
    plotGraphStats(G)
    G = prepareGraph(G)
    return G, weights

def prepareGraph(G: Graph) -> Graph:
    """
    Delete self loop.

    Args:
        G (Graph): _description_

    Returns:
        Graph: _description_
    """
    
    print("Deleting selfloops\n")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def randomWeights(data: DataFrame, reuse: bool) -> dict:
    """
    This function takes in a DataFrame as input and generates a dictionary of random weights for
    each column in the data. It also includes an option to reuse the same random seed.
    
    Parameters:
        - data (DataFrame): Input data for generating weights
        - reuse (bool): Boolean argument for reusing the same random seed. Default is False

    Returns:
        dict: A dictionary containing the generated weights for each column in the input data

    """
    if reuse:
        random.seed(42)
    weights = {key:random.uniform(0, 2) for key in list(data.columns)}
    weights["Name"] = 0
    # weights["Alc"] = 1
    weights["Dalc"] = 1
    weights["Walc"] = 1
    weights["alc"] = 1
    # Changer 1 par 2
    
    return weights

def louvainPartitioning(G: Graph) -> Tuple[dict, Graph]:
    """
    This function takes in a Graph object as input and applies the Louvain community detection algorithm on the graph.
    It returns a dictionary containing the partitioning of the graph's nodes and also prints the number of partitions.
    
    Parameters:
        - G (Graph): The input graph to perform the partitioning on

    Returns:
        dict: A dictionary containing the partitioning of the input graph's nodes
    """
    
    partition = community_louvain.best_partition(G, random_state=42)
    print(f"\nNumber of partitions: {len(set(partition.values()))}")
    
    nx.set_node_attributes(G, partition, 'partition')
    return partition, G

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
    """
    This function takes in a DataFrame and an ndarray as input and runs a loop for a specified number of epochs to
    generate random weights for the input data and apply the Louvain community detection algorithm on the resulting graph.
    It also includes options to save the best weights and score, and prints the execution time for each epoch.
    
    Parameters:
        - data (DataFrame): Input data for generating weights
        - data_vec (ndarray): Array of data vectors for adding edges to the graph
        - epoch (int): Number of times to run the weight generation and partitioning. Default is 10

    Returns:
        Tuple[dict, ndarray]: A dictionary containing the generated weights for each epoch and an array of the scores for each epoch
    """
    
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
                saveWeights(weights, "./weights", "weights_random.yaml")
                saveScore(max(score_list), "best_score_random.txt")
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

def saveXGBScore(score: float):
    
    try:
        best_saved_score = float(loadScore("./weights/best_score_xgboost.txt"))
        print(f"Best saved score: {best_saved_score}")
    except:
        print("Saved score not found, will be created")
    
    # try: 
        # if best_saved_score < score:
            # saveScore(modularity, "best_score_xgboost.txt")
    # except:
        # saveScore(modularity, "best_score_xgboost.txt")
    saveScore(modularity, "best_score_xgboost.txt")

if __name__ == "__main__":
    
    start_time = time.time()
    
    optimize, epoch, graph, xgbweights = parseArguments()
    
    
    data, data_vec = retrieveData(DATA_PATH)
    
    if optimize:
        if xgbweights:
            XGBoostWeightsOptimizer(data)
        else:
            randomWeightOptimizer(data, data_vec, epoch)
    
    if graph:
        G = nx.Graph()
        G, weights = createGraph(G, data, data_vec, random_weights=False, xbgweights=xgbweights)
        graphPlot(G)
        
        partition, G = louvainPartitioning(G)
        plotGraphWithPartition(G, partition)

        part_by_com = refactoringPartition(partition)
        modularity = louvain_community_quality(G, part_by_com)
        
        if xgbweights:
            saveXGBScore(modularity)

        data = addPartitionToData(data, partition)
        saveDFToCSV(data, xbgweights=xgbweights)
        
        saveGraphGephi(G, xgbweights)

        
    delta_time = time.time() - start_time
    print(f"Execution time: {time.strftime('%H:%M:%S', time.gmtime(delta_time))}")