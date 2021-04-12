from typing import Dict,Any,Union,Tuple,List
import tensorflow as tf
import tf2_gnn

def main():

    NodeNumberList=tf.constant([8,4,7], shape=[ 3,])
    NodeNumberList = tf.matrix_inverse(NodeNumberList)
    print(NodeNumberList)

main()