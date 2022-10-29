"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
from re import search
import sys
import math
import warnings
import classes
import search
import numpy as np
import pandas as pd
from data.data import process_data, get_scats_list, process_node
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

def time_to_interval(time):
    split = time.split()
    hms = split[1].split(":")
    min = int(hms[1])/(60)
    time =int(hms[0]) + min
    return time/.25

def plot_results(y_true, y_preds, names, scat):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """

    d = '2006-1-10 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)
    ax.set_title(scat)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.savefig('foo.png')
    plt.show()
   

def initialise_map(file):
    nodes = {}
    for scat in get_scats_list(file):
        node_data = process_node(file, scat)
        nodes[str(node_data[0])] = classes.Node(node_data[0], node_data[1], node_data[2], node_data[3])

    for scat in get_scats_list(file):
        for connection in nodes[str(scat)].get_connections()[2]:
            nodes[str(scat)].add_adjNode(nodes[str(connection)])
        
    return nodes

def get_Flow(y_preds, time):
    itr = time_to_interval(time)
    arr = []
    for i in range(96):
        arr.append(i)
    return np.interp(itr, arr, y_preds)


def findFlowForSearch(scat, time):
    biDir = 'model/biDir/'
    models = [biDir]
    names = ['Bidirectional']
    scats = []
    scats.append(scat)
    # scats = [2000]
    # time = '2006-1-10 13:00'
    
    for num in tqdm(scats):
        
        _, _, X_test, y_test, scaler = process_data('data/data1.xls', 'data/data1.xls', 12, num)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]  
        y_preds = []
        for name, model in zip(names, models):
            
            if name == 'SAEs' or name == "SimpleRNN":
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                
            mdl = load_model(model + str(num) + '.h5')
            predicted = mdl.predict(X_test)
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:96])
            
        return get_Flow(y_preds, time)

def main(argv):
    lstm = 'model/lstm/'
    gru = 'model/gru/'
    saes = 'model/saes/'
    srnn = 'model/srnn/'
    biDir = 'model/biDir/'

    # models = [lstm, gru, saes, srnn, biDir]
    # names = ['LSTM', 'GRU', 'SAEs', 'SimpleRNN', 'Bidirectional']
    # models = argv[1]
    # names = argv[2]
    # scats = argv[3]
    
    ####################### Arguments to pass to get flow value %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    models = [srnn]
    names = ['SimpleRNN']
    scats = [2000]
    time = '2006-1-10 13:00'
    ########################
    
    lag = 12
    file1 = 'data/data1.xls'
    file2 = 'data/data1.xls'
 
    

    
    for num in tqdm(scats):
        
        _, _, X_test, y_test, scaler = process_data(file1, file2, lag, num)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
        
        y_preds = []
        for name, model in zip(names, models):
            if name == 'SAEs' or name == "SimpleRNN":
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            file = 'images/' + name + '.png'
            mdl = load_model(model + str(num) + '.h5')
            plot_model(mdl, to_file=file, show_shapes=True)
            predicted = mdl.predict(X_test)
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:96])
            print(name)
            eva_regress(y_test, predicted)
        
    
        plot_results(y_test[: 96], y_preds, names, num)
        
    
    #


if __name__ == '__main__':
    # main(sys.argv)
    search.harrisonsMethod(970, 4040, '2006-1-10 13:00')
