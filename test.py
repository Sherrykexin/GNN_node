import glob
import json
import os
from typing import Any, Dict
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import seaborn
import tensorflow as tf
import tf2_gnn
from tf2_gnn.cli_utils.training_utils import train, log_line, make_run_id
from tf2_gnn.data import DataFold, HornGraphSample, HornGraphDataset
from tf2_gnn.models import InvariantArgumentSelectionTask, InvariantNodeIdentifyTask

from Miscellaneous import pickleWrite, pickleRead, drawBinaryLabelPieChart
from dotToGraphInfo import parseArgumentsFromJson
from utils import plot_confusion_matrix,get_recall_and_precision,plot_ROC,assemble_name
def main():
    ss = ["predicateName_0","predicateArgument_0","predicateArgument_1","FALSE","clause_0","!","=","symbolicConstant_0","0",">=","symbolicConstant_1","1","=","+","symbolicConstant_2","*","-1","clauseHead_0","clauseArgument_0","clauseArgument_1","clause_1",">=","symbolicConstant_3","=","symbolicConstant_4","clauseHead_1","clauseArgument_2","clauseArgument_3","clause_2",">=","symbolicConstant_5","clauseHead_2","clauseBody_0","clauseArgument_4","clauseArgument_5","3",">=",">=",">=",">=","+","*","*",">=","+",">=","=","+",">=","=",">=","=",">=",">=",">=",">=",">=",">=","+",">=","+",">=","=","+",">=","+","=","+",">=","+","=","+",">=","=",">=","=",">=","=","=",">=","=",">=","=",">="]
    aa =  [25, 23, 24, 8, 19, 0, 6, 26, 4, 7, 27, 5, 6, 2, 28, 1, 3, 16, 9, 10, 20, 7, 29, 6, 30, 17, 11, 12, 21, 7, 31, 18, 15, 13, 14, 22, 7, 7, 7, 7, 2, 1, 1, 7, 2, 7, 6, 2, 7, 6, 7, 6, 7, 7, 7, 7, 7, 7, 2, 7, 2, 7, 6, 2, 7, 2, 6, 2, 7, 2, 6, 2, 7, 6, 7, 6, 7, 6, 6, 7, 6, 7, 6, 7]
    print(len(aa))
main()
