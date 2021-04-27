from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
from tf2_gnn.layers.gnn import GNNInput
from tf2_gnn.models.graph_binary_classification_task import GraphBinaryClassificationTask
from invariantArgumentSelectionModel import InvariantArgumentSelectionModel
from tf2_gnn.models import InvariantArgumentSelectionTask, InvariantNodeIdentifyTask
from horn_dataset import train_on_graphs
import json
import numpy as np
from tf2_gnn.data import DataFold, HornGraphSample, HornGraphDataset


graphs_node_label_ids= []
graphs_adjacency_lists = []
graphs_templateIndices =[]
graph_templateRelevanceLabel = []
total_number_of_node = 0

def read_json(file_path):
    
    with open(file_path) as f:
        loaded_graph = json.load(f)
        graphs_node_label_ids.append(loaded_graph["nodeIds"])  #lists of node id
        graphs_adjacency_lists.append(loaded_graph["binaryAdjacentList"]) #lists of adjacency lists
        graphs_templateIndices.append(loaded_graph["templateIndices"])
        graph_templateRelevanceLabel.append(loaded_graph["templateRelevanceLabel"])
        total_number_of_nodes = len(loaded_graph["nodeIds"])
        return total_number_of_nodes
    


  

    
def main():
    
    parameter_list = []
    label_list=[]
    label_list.append("template_relevance")
    force_read = True
    form_label = True
    file_type = ".smt2"
    GPU=True
    pickle = True
    benchmark_name = "single-layer-graph-example/"

    hyper_parameters={"nodeFeatureDim":64,"num_layers":12,"regression_hidden_layer_size":[64,32,16],"threshold":0.5,"max_nodes_per_batch":1000}
    for label in label_list:
        parameter_list.append(
            parameters(relative_path="/Users/sherry/Desktop/extractable-three-fold-lin+nonlin/train_data/"+benchmark_name,
                       absolute_path="/Users/sherry/Downloads/Systematic-Predicate-Abstraction-using-Machine-Learning-master/Heuristic_selection/benchmarks/",
                       json_type=".mono-layerHornGraph.JSON", label=label))


    for param in parameter_list:
        if pickle==False:
            train_on_graphs(benchmark_name=param.absolute_path[param.absolute_path.find("/benchmarks/")+len("/benchmarks/"):-1], label=param.label, force_read=force_read,
                            train_n_times=1,path=param.absolute_path, file_type=file_type, form_label=form_label,
                            json_type=param.json_type,GPU=GPU,pickle=pickle,hyper_parameters=hyper_parameters)
        else:
            train_on_graphs(benchmark_name=param.benchmark_name(),
                            label=param.label, force_read=force_read,
                            train_n_times=1, path=param.relative_path, file_type=file_type, form_label=form_label,
                            json_type=param.json_type, GPU=GPU, pickle=pickle,hyper_parameters=hyper_parameters)
        tf.keras.backend.clear_session()
   
    
    
    
    final_graphs=[]
    final_graphs.append(
            HornGraphSample(
                adjacency_lists=graphs_node_label_ids,
                node_features=np.array(graphs_node_label_ids),
                node_indices=np.array(node_indices),
                node_label=np.array(learning_labels)
            )
        )
"""     inputs={'NodeNumberList':graphs_node_label_ids, #[97]
            'node_to_graph_map':node_to_graph_map,
            'node_label_ids':node_label_ids, #[0, 1, 2,..., 97]
            'num_edge_types':len(adjacency_lists), # 97
            'adjacency_lists':adjacency_lists}
"""
    # params = tf2_gnn.GNN.get_default_hyperparameters()
    # params["hidden_dim"] = 8
    # layer = InvariantArgumentSelectionModel(params)
    # output = layer(input) 
    # print(output)
 

class parameters():
    def __init__(self, relative_path,absolute_path,json_type,label):
        self.relative_path=relative_path
        self.absolute_path=absolute_path
        self.json_type=json_type
        self.label=label
    def benchmark_name(self):
        return self.absolute_path[self.absolute_path.find("/benchmarks/") + len("/benchmarks/"):-1]

main()