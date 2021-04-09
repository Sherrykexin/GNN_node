from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
from tf2_gnn.layers.gnn import GNNInput
from tf2_gnn.models.graph_binary_classification_task import GraphBinaryClassificationTask
import json

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
        loaded_graph = json.read(f)
        graphs_node_label_ids.append(loaded_graph["nodeIds"])   
        print (graphs_node_label_ids)





def main():
     
     input = GNNInput(
         node_features = tf.random.normal(shape=(4, 3)),
         adjacency_lists = (
             tf.constant([[0, 1], [1, 2]], dtype=tf.int32),
             ),
         node_to_graph_map = tf.fill(dims=(4,), value=0),
         num_graphs = 1,
         )
    
     params = tf2_gnn.GNN.get_default_hyperparameters()
     params["hidden_dim"] = 8
     layer = tf2_gnn.GNN(params)
     output = layer(input)
     read_json()
     #print(output)
 


main()