from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
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

    nodeFeatureDim=3
    file_path= "04.c_000.smt2.hybrid-layerHornGraph.JSON"
    total_number_of_node= read_json(file_path)
    NodeNumberList=total_number_of_node
    numberOfNode=total_number_of_node #total nodes in three graph

    node_label_ids = tf.constant(list(range(0, numberOfNode))) #[0, 1, 2,..., 14]


    #get node_to_graph_map from NodeNumberList
    node_to_graph_map=[]
    #for i, nodeNumber in enumerate(NodeNumberList):
        #node_to_graph_map.append(tf.fill(dims=(nodeNumber,), value=i))
    node_to_graph_map=tf.fill(dims=(numberOfNode,), value=0)

    adjacency_lists = graphs_adjacency_lists


    parameters = tf2_gnn.GNN.get_default_hyperparameters()
    parameters["hidden_dim"] = 4
    parameters["num_layers"]= 1
    parameters['node_label_vocab_size']=numberOfNode
    parameters['node_label_embedding_size']=nodeFeatureDim
    parameters['num_edge_types']=len(adjacency_lists)


    inputs={'NodeNumberList':NodeNumberList, #[8,4,3]
            'node_to_graph_map':node_to_graph_map,
            'node_label_ids':node_label_ids, #[0, 1, 2,..., 14]
            'num_edge_types':len(adjacency_lists), # 3
            'adjacency_lists':adjacency_lists}
    for edge_type_idx,edgeType in enumerate(adjacency_lists): # 3,4,2
        inputs[f"adjacency_list_{edge_type_idx}"]=tf.TensorSpec(shape=(None, edgeType.shape[1]), dtype=tf.int32)

    layers=InvariantArgumentSelectionModel(parameters)
    output=layers(inputs)
    print(output)



main()