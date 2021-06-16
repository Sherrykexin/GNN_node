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
from matplotlib.colors import from_levels_and_colors
def main():
    matrix = [[0.1,0.2,0.1,-0.1],[-0.1,-0.2,0.11,0.144]]
    draw_pixel_graph(matrix)



def draw_pixel_graph(matrix):
    #draw_train_weights2 = [j["regression_weights"] for j in weights_list_train]
    matrix = np.array(matrix)
    num_levels = 5
    vmin, vmax = matrix.min(), matrix.max()
    print("debug vmin,vmax:", vmin, vmax)
    midpoint = 0
    levels = np.linspace(vmin, vmax, num_levels)
    midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
    vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.2, 0.4])
    print("val", vals)
    colors = plt.cm.seismic(vals)
    cmap, norm = from_levels_and_colors(levels, colors)
    plt.imshow(matrix, cmap=cmap,interpolation='none' )
        #ax.set_xticks([])
        #ax.set_yticks([])
    plt.colorbar()
    
    plt.show()


main()

