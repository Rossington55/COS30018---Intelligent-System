"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_node, get_scats_list
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

def main():

    file1 = 'data/data1.xls'

    for scat in get_scats_list(file1):
        node = process_node(file1, scat[0])
        print('SCAT : ' + str(node[0]))
        print('Longitude : ' + str(node[1]))
        print('Latitude : ' + str(node[2]))
        print('Street Connections : ')
        for connection in node[3]:
            print('Street Name: ' + connection[0])
            print('Direction: ' + connection[1])
            print('Connected SCAT: ' + str(connection[2]))
            print('===================================')

        


if __name__ == '__main__':
    main()
