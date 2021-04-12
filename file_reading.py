from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
from tf2_gnn.layers.gnn import GNNInput
from tf2_gnn.models.graph_binary_classification_task import GraphBinaryClassificationTask
import json
import numpy as np

graphs_node_label_ids= []
graphs_node_symbols = []
graphs_argument_indices = []
graphs_adjacency_lists = []
graphs_argument_scores = []
parsed_arguments = []
graphs_control_location_indices = []
graphs_label_indices = []
graphs_learning_labels = []
total_number_of_node = 0

file_name_list = []
skipped_file_list=[]

def read_json():
    with open('04.c_000.smt2.hybrid-layerHornGraph.JSON') as f:
        loaded_graph = json.load(f)
        graphs_node_label_ids.append(loaded_graph["nodeIds"])  #lists of node id
        graphs_adjacency_lists.append(loaded_graph["binaryAdjacentList"]) #lists of adjacency lists
        total_number_of_node = len(loaded_graph["nodeIds"])
        #print (graphs_node_label_ids)
        #print("length is " ,total_number_of_node)
         

def main():
    read_json()
    node_to_graph_map = []
    for i, total_number_of_node in enumerate(graphs_node_label_ids):
        node_to_graph_map.append(tf.fill(dims = (total_number_of_node,), value = i))
    node_to_graph_map=tf.concat(node_to_graph_map, 0)

    

    
    input = GNNInput(
       
         node_features = (graphs_node_label_ids, 3),
         adjacency_lists = graphs_adjacency_lists,
         node_to_graph_map = node_to_graph_map,
         num_graphs = 1,
         )
    
    params = tf2_gnn.GNN.get_default_hyperparameters()
    params["hidden_dim"] = 8
    layer = tf2_gnn.GNN(params)
    output = layer(input) 
    print(output)
 


main()