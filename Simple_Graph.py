from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn
from tf2_gnn.layers.gnn import GNNInput
from tf2_gnn.models.graph_binary_classification_task import GraphBinaryClassificationTask

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
     print(output)
 


main()