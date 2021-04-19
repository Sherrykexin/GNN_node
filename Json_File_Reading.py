from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
from tf2_gnn.layers.gnn import GNNInput
from tf2_gnn.models.graph_binary_classification_task import GraphBinaryClassificationTask
from invariantArgumentSelectionModel import InvariantArgumentSelectionModel
import json
import numpy as np

graphs_node_label_ids= []
graphs_adjacency_lists = []
total_number_of_node = 0

def read_json(file_path):
    
    with open(file_path) as f:
        loaded_graph = json.load(f)
        graphs_node_label_ids.append(loaded_graph["nodeIds"])  #lists of node id
        graphs_adjacency_lists.append(loaded_graph["binaryAdjacentList"]) #lists of adjacency lists
        total_number_of_node = len(loaded_graph["nodeIds"])
        return total_number_of_node

def main():
    file_path= "04.c_000.smt2.hybrid-layerHornGraph.JSON"
    total_number_of_node= read_json(file_path)
    
    inputs={'NodeNumberList':graphs_node_label_ids, #[97]
            'node_to_graph_map':node_to_graph_map,
            'node_label_ids':node_label_ids, #[0, 1, 2,..., 97]
            'num_edge_types':len(adjacency_lists), # 97
            'adjacency_lists':adjacency_lists}
    
    params = tf2_gnn.GNN.get_default_hyperparameters()
    params["hidden_dim"] = 8
    layer = InvariantArgumentSelectionModel(params)
    output = layer(input) 
    print(output)
 


main()