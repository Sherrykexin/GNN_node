import tf2_gnn
from typing import Dict,Any
import tensorflow as tf

class InvariantArgumentSelectionModel(tf.keras.Model):
    def __init__(self, params: Dict[str, Any],  **kwargs):
        super().__init__(**kwargs)
        self._params = params
        self._embedding_layer = tf.keras.layers.Embedding(
            input_dim=params["node_label_vocab_size"],
            output_dim=params["node_label_embedding_size"]
        )
        self._num_edge_types =params['num_edge_types']
        self._gnn = tf2_gnn.GNN(params)

    def build(self, input_shapes):
        # build node embedding layer
        self._embedding_layer.build(tf.TensorShape((None,)))
        # build gnn layers
        self._gnn.build(
            tf2_gnn.GNNInput(
                node_features=tf.TensorShape((None, self._params["node_label_embedding_size"])),
                adjacency_lists=tuple(
                    input_shapes[f"adjacency_list_{edge_type_idx}"]
                    for edge_type_idx in range(self._num_edge_types)
                ),
                node_to_graph_map=tf.TensorShape((None,)),
                num_graphs=tf.TensorShape(()),
            )
        )
        # todo: build task-specific layer
        super().build([])


    def call(self, inputs, training: bool = False):
        # call node embedding layer
        node_labels_embedded = self._embedding_layer(inputs["node_label_ids"], training=training)
        # call gnn and get graph representation
        gnn_input = tf2_gnn.GNNInput(
            node_features=node_labels_embedded,
            adjacency_lists=inputs['adjacency_lists'],
            # Here, the adjacency lists would need to be extracted from the input, which is easiest as Dict[str, tf.Tensor/np.ndarray]
            node_to_graph_map=inputs['node_to_graph_map'],
            num_graphs=len(inputs['NodeNumberList']),
        )
        final_node_representations = self._gnn(gnn_input, training=training)

        # self.compute_task_output(inputs, final_node_representations, training)
        # todo: Here, predictions should be made, based on the node representations

        return final_node_representations

